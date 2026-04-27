from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gen_eval.schemas import GenerationSample


@dataclass(slots=True)
class _RuntimeBundle:
    torch: Any
    functional: Any
    dino_model: Any
    image_transform: Any
    load_video: Any
    device: str


class SubjectConsistency:
    name = "subject_consistency"

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self._runtime: _RuntimeBundle | None = None

    def evaluate(self, samples: list[GenerationSample]) -> dict[str, Any]:
        paired_samples = [
            sample for sample in samples if sample.generated_video and sample.reference_video
        ]
        if not paired_samples:
            return self._result(
                score=None,
                num_samples=0,
                details={"evaluated_samples": []},
                status="skipped",
                reason="Subject consistency requires samples with both generated_video and reference_video.",
            )

        runtime, reason = self._get_runtime()
        if runtime is None:
            return self._result(
                score=None,
                num_samples=0,
                details={"evaluated_samples": []},
                status="skipped",
                reason=reason,
            )

        video_results = []
        skipped_samples = []
        failed_samples = []
        total_video_sim = 0.0
        total_transition_count = 0
        total_tji_score = 0.0
        total_ts = 0.0

        for sample in paired_samples:
            try:
                sample_result = self._calculate_sample(sample, runtime)
            except FileNotFoundError as exc:
                skipped_samples.append({"sample_id": sample.sample_id, "reason": str(exc)})
                continue
            except Exception as exc:  # pragma: no cover - defensive runtime guard
                failed_samples.append({"sample_id": sample.sample_id, "reason": str(exc)})
                continue

            video_results.append(sample_result)
            total_video_sim += sample_result["video_sim"]
            total_transition_count += sample_result["cnt_per_video"]
            total_tji_score += sample_result["TJI_score"]
            total_ts += sample_result["ts"]

        if not video_results:
            status = "failed" if failed_samples else "skipped"
            reason = (
                "All paired samples failed during subject consistency evaluation."
                if failed_samples
                else "No paired samples could be evaluated."
            )
            return self._result(
                score=None,
                num_samples=0,
                details={
                    "evaluated_samples": [],
                    "skipped_samples": skipped_samples,
                    "failed_samples": failed_samples,
                },
                status=status,
                reason=reason,
            )

        sim_per_frame = (
            total_video_sim / total_transition_count if total_transition_count else 0.0
        )
        tji_per_video = total_tji_score / len(video_results)
        ts_per_video = total_ts / len(video_results)

        return self._result(
            score=ts_per_video,
            num_samples=len(video_results),
            details={
                "subject_consistency_per_frame": sim_per_frame,
                "tji_per_video": tji_per_video,
                "ts_per_video": ts_per_video,
                "evaluated_samples": video_results,
                "skipped_samples": skipped_samples,
                "failed_samples": failed_samples,
            },
            status="ok",
            reason=None,
        )

    def _get_runtime(self) -> tuple[_RuntimeBundle | None, str | None]:
        if self._runtime is not None:
            return self._runtime, None

        try:
            import imageio.v2 as imageio  # type: ignore
            import torch
            from torchvision import transforms
            from torchvision.transforms import Compose, Normalize, Resize
        except Exception as exc:
            return None, f"Required runtime dependencies are unavailable: {exc}"

        repo_or_dir = self.config.get("repo_or_dir")
        weights_path = self.config.get("weights_path")
        model_name = self.config.get("model_name", "dino_vitb16")
        device = self.config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")

        if not repo_or_dir:
            return None, "DINO runtime requires a local 'repo_or_dir' in metric config."
        if not weights_path:
            return None, "DINO runtime requires a local 'weights_path' in metric config."
        if not Path(repo_or_dir).exists():
            return None, f"DINO repo_or_dir does not exist: {repo_or_dir}"
        if not Path(weights_path).exists():
            return None, f"DINO weights_path does not exist: {weights_path}"

        load_dict = {
            "repo_or_dir": str(repo_or_dir),
            "model": str(model_name),
            "path": str(weights_path),
            "source": "local",
        }

        try:
            dino_model = torch.hub.load(**load_dict).to(device)
            dino_model.eval()
        except Exception as exc:
            return None, f"DINO model could not be loaded during evaluation: {exc}"

        image_transform = Compose(
            [
                Resize(size=224, antialias=False),
                transforms.Lambda(lambda x: x.float().div(255.0)),
                Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        def load_video(video_path: str) -> Any:
            frames = imageio.mimread(video_path)
            if not frames:
                raise ValueError(f"No frames could be read from video: {video_path}")
            frame_tensor = torch.stack(
                [torch.from_numpy(frame).permute(2, 0, 1) for frame in frames],
                dim=0,
            )
            return frame_tensor

        self._runtime = _RuntimeBundle(
            torch=torch,
            functional=torch.nn.functional,
            dino_model=dino_model,
            image_transform=image_transform,
            load_video=load_video,
            device=device,
        )
        return self._runtime, None

    def _calculate_sample(
        self, sample: GenerationSample, runtime: _RuntimeBundle
    ) -> dict[str, Any]:
        generated_path = self._require_path(sample.generated_video, "generated_video", sample.sample_id)
        reference_path = self._require_path(sample.reference_video, "reference_video", sample.sample_id)

        image_features = self._get_video_image_feature(generated_path, runtime)
        video_sim, sim_per_image, cnt_per_video = self._calculate_acm(image_features, runtime)
        tji, tji_score = self._calculate_tji(image_features, runtime)

        reference_features = self._get_video_image_feature(reference_path, runtime)
        _, ref_sim_per_image, _ = self._calculate_acm(reference_features, runtime)
        ref_tji, _ = self._calculate_tji(reference_features, runtime)

        s_acm = self._rel_score(sim_per_image, ref_sim_per_image, runtime)
        s_tji = self._rel_score(tji, ref_tji, runtime)
        _, mrs = self._motion_align(image_features, reference_features, runtime, beta=0.5)
        ts = sim_per_image * (s_acm * s_tji).sqrt() * mrs.sqrt()
        if runtime.torch.isnan(ts):
            ts = runtime.torch.tensor(0.0, device=ts.device)

        return {
            "sample_id": sample.sample_id,
            "generated_video": generated_path,
            "reference_video": reference_path,
            "video_results": sim_per_image,
            "video_sim": video_sim,
            "cnt_per_video": cnt_per_video,
            "TJI": tji.item(),
            "TJI_score": tji_score.item(),
            "ts": ts.item(),
        }

    def _get_video_image_feature(self, video_path: str, runtime: _RuntimeBundle) -> Any:
        images = runtime.load_video(video_path)
        images = runtime.image_transform(images)
        images = images.to(runtime.device)
        with runtime.torch.no_grad():
            image_features = runtime.dino_model(images)
            image_features = runtime.functional.normalize(image_features, dim=-1, p=2)
        return image_features

    def _calculate_acm(self, image_features: Any, runtime: _RuntimeBundle) -> tuple[float, float, int]:
        if len(image_features) < 2:
            raise ValueError("Subject consistency requires at least 2 frames per video.")

        video_sim = 0.0
        cnt_per_video = 0
        former_image_feature = None
        first_image_feature = None

        for i in range(len(image_features)):
            image_feature = image_features[i].unsqueeze(0)
            if i == 0:
                first_image_feature = image_feature
                former_image_feature = image_feature
                continue

            sim_pre = max(
                0.0, runtime.functional.cosine_similarity(former_image_feature, image_feature).item()
            )
            sim_fir = max(
                0.0, runtime.functional.cosine_similarity(first_image_feature, image_feature).item()
            )
            video_sim += (sim_pre + sim_fir) / 2
            cnt_per_video += 1
            former_image_feature = image_feature

        sim_per_image = video_sim / (len(image_features) - 1)
        return video_sim, sim_per_image, cnt_per_video

    def _calculate_tji(self, image_features: Any, runtime: _RuntimeBundle) -> tuple[Any, Any]:
        if len(image_features) < 3:
            raise ValueError("Subject consistency requires at least 3 frames per video.")

        v = (image_features[1:] - image_features[:-1]).norm(dim=1)
        a = (image_features[2:] - 2 * image_features[1:-1] + image_features[:-2]).norm(dim=1)
        tji = (a / (0.5 * (v[1:] + v[:-1]) + 1e-8)).mean()
        tji_score = runtime.torch.exp(-0.5 * tji)
        return tji, tji_score

    def _rel_score(self, x_gen: Any, x_gt: Any, runtime: _RuntimeBundle, w: float = 4.0) -> Any:
        x_gen_tensor = runtime.torch.as_tensor(x_gen, device=runtime.device)
        x_gt_tensor = runtime.torch.as_tensor(x_gt, device=runtime.device)
        return runtime.torch.exp(
            -w * runtime.torch.abs(runtime.torch.log((x_gen_tensor + 1e-8) / (x_gt_tensor + 1e-8)))
        )

    def _motion_align(
        self, features_a: Any, features_b: Any, runtime: _RuntimeBundle, beta: float = 3.0
    ) -> tuple[Any, Any]:
        v_a = features_a[1:] - features_a[:-1]
        v_b = features_b[1:] - features_b[:-1]
        v_a_n = v_a / (v_a.norm(dim=1, keepdim=True) + 1e-8)
        v_b_n = v_b / (v_b.norm(dim=1, keepdim=True) + 1e-8)
        mda = (v_a_n * v_b_n).sum(1).mean().clamp(-1, 1)
        s_a = v_a.norm(dim=1)
        s_b = v_b.norm(dim=1)
        mrs = runtime.torch.exp(
            -beta * runtime.torch.mean(runtime.torch.abs(runtime.torch.log((s_a + 1e-8) / (s_b + 1e-8))))
        )
        return mda, mrs

    def _require_path(self, raw_path: str | None, field_name: str, sample_id: str) -> str:
        if not raw_path:
            raise FileNotFoundError(f"Sample '{sample_id}' is missing required field '{field_name}'.")
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Sample '{sample_id}' {field_name} does not exist: {raw_path}"
            )
        return str(path)

    def _result(
        self,
        *,
        score: float | None,
        num_samples: int,
        details: dict[str, Any],
        status: str,
        reason: str | None,
    ) -> dict[str, Any]:
        result = {
            "metric": self.name,
            "score": score,
            "num_samples": num_samples,
            "details": details,
            "status": status,
        }
        if reason is not None:
            result["reason"] = reason
        return result

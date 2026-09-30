# Third-party components

- [OpenCV Zoo LaMa ONNX model](https://huggingface.co/opencv/inpainting_lama): Apache License. The expected file is `inpainting_lama_2025jan.onnx` (SHA-256 `7df918ac3921d3daf0aae1d219776cf0dc4e4935f035af81841b40adcf74fdf2`).
- [FFmpeg 9.0.1 Windows essentials build by Gyan](https://www.gyan.dev/ffmpeg/builds/): its included `LICENSE` identifies GPL v3. The included `README.txt` points to the matching [FFmpeg source revision](https://github.com/FFmpeg/FFmpeg/commit/bf1b838f2a). Include those files when distributing the binary.
- Python packages and transitive dependencies are listed in `requirements.txt` and retain their own licenses.
- Optional CUDA processing uses the separately installed [simple-lama-inpainting](https://github.com/enesmsahin/simple-lama-inpainting) package and a user-supplied Big-LaMa model. The code and model files are not bundled with this repository or its portable package. See the [original LaMa project](https://github.com/advimman/lama) for model provenance and applicable terms.

The LaMa model and FFmpeg are separate executables/data components. This repository does not track their binary files. Local builds may place them in `third_party` or include them in a release package with their license notices.

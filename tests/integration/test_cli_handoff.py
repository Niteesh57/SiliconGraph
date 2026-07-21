"""Tests for assets handed from the Python orchestration layer to C++."""

from armcc.cli import _buildCppCompileCommand


def test_cpp_compile_command_includes_generated_package_assets(tmp_path):
    command = _buildCppCompileCommand(
        armcc_bin="armcc-compiler",
        graph_json="C:/work/exported_graph.json",
        targets="generic_arm64",
        quantization="int8",
        context_lengths="128,512",
        work_dir=str(tmp_path),
        output_path="C:/out/model.armpack",
        calibration_path="C:/work/calibration.json",
        verbose=True,
    )

    assert command == [
        "armcc-compiler", "compile",
        "--graph", "C:/work/exported_graph.json",
        "--targets", "generic_arm64",
        "--quant", "int8",
        "--context-lengths", "128,512",
        "--tokenizer-dir", str(tmp_path / "tokenizer"),
        "--runtime-config", str(tmp_path / "runtime_config.json"),
        "--output", "C:/out/model.armpack",
        "--calibration", "C:/work/calibration.json",
        "--verbose",
    ]

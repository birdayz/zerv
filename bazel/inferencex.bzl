"""Reviewed, hash-pinned upstream benchmark client only; never a production dependency."""

_FILES = {
    "__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "benchmark_serving.py": "f98529033750e38763603cc9c83bcebfd364cf58f9fd5e867eec9dcdde0e7a44",
    "backend_request_func.py": "b8c410db4036eac4f3b2bae1d08d32b9627acae70a3d83c2acbff0df642a4a3f",
    "benchmark_outcome.py": "2d5052eebce4eeffdb85f8ef5cb6c59e6124c834115f6fbe49be55d583de3e71",
    "benchmark_utils.py": "74d521852a946828b4805ad0939d28f10d9ce35ae2e081032d080114123e881d",
    "encoding_dsv4.py": "9509c39bf09ad0767fb4094047c95e85dd390b7657da658cf3e7ee623004970f",
}

_RESULTS = {
    "__init__.py": "0ed9dde418ac02031463a9527169522f48225ef6f04adafbeb8dd83bfcc49834",
    "fixed_sequence.py": "d6ec6bbc8ddde9da1f68cba78f260e6dcec39c1e3bb3409eebc6a6dd8fe672bb",
    "collect_results.py": "5fa3636bb962d303d2f2b2553dcda4a753e353ce741fe52f0070255fa6ca9ea6",
    "metadata.py": "249ceefe318eead96d7442ae070abae1efcd63c5246e34bc00fac6ccf6355b9e",
    "topology.py": "0020502c30674a82a1a9e4d731329b9256f5b2a980efaf77bedccde1c1eb2684",
    "power/__init__.py": "1e0ab125c218170612f9e78cf5de03849568676e757297bd06465f29ec841d96",
    "power/audit.py": "8c74cc4a4b2de79b02aa2cc6a2de85803361470b8d4edda68746f8bfbabcb656",
    "power/common.py": "df24e9f5b7cccd6514f549f9787482ead24cc853286dc5b991f3645b11e3184f",
    "power/single_node.py": "d6a855c36ec1f7337b497842fcff7f930a610aec031fdf5855de3d8acc10886d",
    "power/window.py": "5f912ac11bf1d84aa743fd1813a99798696408b14a969f9c48cc4dceaea884d6",
    "power/multinode.py": "60ad241ff58953347c96ab41a326bdd30d080fb5d1b308b9a0c3ed2f83002695",
    "power/native_multinode.py": "a2dac358d841a02dd0bd097c2a0bb66b1dfbe68bb949387a886cc9bdcd573023",
}

def _source(ctx):
    base = "https://raw.githubusercontent.com/SemiAnalysisAI/InferenceX/f437f7bfd164422036b0de7e3818f8afb5bc70d7/inferencex-e2e/infx/bench_serving/"
    for name, sha in _FILES.items():
        ctx.download(base + name, "infx/bench_serving/" + name, sha256 = sha)
    results_base = "https://raw.githubusercontent.com/SemiAnalysisAI/InferenceX/f437f7bfd164422036b0de7e3818f8afb5bc70d7/inferencex-e2e/infx/results/"
    for name, sha in _RESULTS.items():
        ctx.download(results_base + name, "infx/results/" + name, sha256 = sha)
    ctx.file("infx/__init__.py", "")
    ctx.file("BUILD.bazel", """load("@rules_python//python:py_library.bzl", "py_library")
py_library(name="client", srcs=["infx/__init__.py"] + glob(["infx/bench_serving/*.py"]), imports=["."], visibility=["//visibility:public"])
py_library(name="results", srcs=glob(["infx/results/**/*.py"]), deps=[":client"], imports=["."], visibility=["//visibility:public"])
""")

inferencex_source = repository_rule(implementation = _source)

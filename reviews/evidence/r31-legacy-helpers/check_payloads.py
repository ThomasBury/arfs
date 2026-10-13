import json
import tarfile
import zipfile
from pathlib import Path

root = Path("/home/bsatom/Documents/arfs")
work = Path("/tmp/arfs-r31")
evidence = root / "reviews/evidence/r31-legacy-helpers"
source = {
    p.relative_to(root / "src").as_posix(): p.read_bytes()
    for p in (root / "src/arfs").rglob("*")
    if p.is_file()
    and p.suffix in {".py", ".joblib", ".zip", ".rst"}
    and "__pycache__" not in p.parts
}
with zipfile.ZipFile(next((work / "dist").glob("*.whl"))) as wheel:
    for name, content in source.items():
        assert wheel.read(name) == content, ("wheel", name)
with tarfile.open(next((work / "dist").glob("*.tar.gz"))) as sdist:
    prefix = sdist.getnames()[0].split("/")[0]
    for name, content in source.items():
        assert sdist.extractfile(prefix + "/src/" + name).read() == content, (
            "sdist",
            name,
        )


result = {
    "source_sdist_wheel_files": len(source),
    "data_payloads": sum(not name.endswith(".py") for name in source),
}
(evidence / "payloads.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))

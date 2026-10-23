"""Ask Claude for a plain-language explanation of a DermNet class (or of the model's prediction for an image).

    python explain.py "Eczema Photos"
    python explain.py --image photo.jpg --arch resnet50   # needs checkpoints/<arch>/best.pt from train.py

Needs `pip install anthropic` and ANTHROPIC_API_KEY set. Educational only, not medical advice.
"""
import argparse
import json
from pathlib import Path

import anthropic

SYSTEM = (
    "You explain skin conditions for a general audience in an educational demo. In under 250 words, cover: "
    "what the condition (or group of conditions) is, common signs, and how it is usually treated. "
    "Do not give drug doses. End by saying a photo classifier can be wrong and a doctor or dermatologist "
    "should confirm any diagnosis and treatment."
)


def predict(image_path, arch):
    import torch
    from PIL import Image

    import train

    classes = json.loads(Path(f"results/{arch}/confusion_matrix.json").read_text())["classes"]
    model = train.build_model(arch, len(classes))
    model.load_state_dict(torch.load(f"checkpoints/{arch}/best.pt", map_location="cpu"))
    tf = train.transforms.Compose([
        train.transforms.Resize(224), train.transforms.CenterCrop(224),
        train.transforms.ToTensor(), train.transforms.Normalize(train.MEAN, train.STD),
    ])
    with torch.no_grad():
        probs = model.eval()(tf(Image.open(image_path).convert("RGB"))[None]).softmax(1)[0]
    i = int(probs.argmax())
    return classes[i], float(probs[i])


def explain(condition):
    response = anthropic.Anthropic().beta.messages.create(
        model="claude-opus-5-5",
        max_tokens=4000,
        output_config={"effort": "low"},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        system=SYSTEM,
        messages=[{"role": "user", "content": f"Explain this DermNet category: {condition}"}],
    )
    if response.stop_reason == "refusal":
        return "The model declined to answer this request."
    return "".join(b.text for b in response.content if b.type == "text")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("condition", nargs="?", help="a DermNet class name")
    ap.add_argument("--image", help="classify this image first, then explain the prediction")
    ap.add_argument("--arch", default="resnet50")
    args = ap.parse_args()
    if args.image:
        args.condition, p = predict(args.image, args.arch)
        print(f"Predicted: {args.condition} ({p:.0%})\n")
    if not args.condition:
        ap.error("give a condition name or --image")
    print(explain(args.condition))


if __name__ == "__main__":
    main()

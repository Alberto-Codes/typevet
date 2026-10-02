"""Face pair: do two face images show the same person?

Scenario: a reviewer has two photographs and asks whether they show the same
person. One typed Noul asks the question over both images. The script prints
``P(yes)``, then the winner on the last line: ``yes`` when the probability is
above 0.5, otherwise ``no``. The model answers about appearance. It makes no
identity or authenticity claim.

Run with: `uv run python examples/face_pair/run.py`

Set ``TYPEVET_EXAMPLE_IMAGE_A`` and ``TYPEVET_EXAMPLE_IMAGE_B`` to two image
files (``.png``, ``.jpg`` or ``.jpeg``). Run the command from the repository
root, because the state and the question come from the ``typevet_evals``
workspace member. ``TYPEVET_BACKEND`` selects the backend (default
``llama_cpp``). For a live llama.cpp run, set
``TYPEVET_LLAMA__MULTIMODAL_MODEL`` to a Gemma 4 vision model id.
``TYPEVET_BACKEND=fake`` runs offline.

Examples:
    ```bash
    export TYPEVET_EXAMPLE_IMAGE_A=a.jpg TYPEVET_EXAMPLE_IMAGE_B=b.jpg
    TYPEVET_BACKEND=fake uv run python examples/face_pair/run.py
    ```

See Also:
    - [typevet.adapters.inbound.open_judgment][]: Session that the backend selects.
    - examples/README.md: Every example and its command.
"""

from __future__ import annotations

import os
from pathlib import Path

from typevet.adapters.inbound import open_judgment
from typevet.domain import ImageInput, Noul

# Repo-only: the loader below builds the request in the evals workspace member.
from typevet_evals.datasets.lfw import LfwFace, LfwPair
from typevet_evals.face_match.request import SAME_PERSON, build_face_match_request

THRESHOLD = 0.5
MIME_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


def read_image(variable: str) -> ImageInput:
    """Read the image file that an environment variable names.

    Args:
        variable: Name of the variable that holds the image path.

    Returns:
        The image bytes with the mime type from the file suffix.

    Raises:
        ValueError: When the variable is not set, the file does not exist or
            the suffix is not ``.png``, ``.jpg`` or ``.jpeg``.
    """
    value = os.environ.get(variable)
    if not value:
        raise ValueError(f"set {variable} to the path of an image file")
    path = Path(value)
    if not path.is_file():
        raise ValueError(f"{variable} names a file that does not exist: {path}")
    mime_type = MIME_TYPES.get(path.suffix.lower())
    if mime_type is None:
        raise ValueError(f"{variable} must name a .png, .jpg or .jpeg file: {path}")
    return ImageInput(data=path.read_bytes(), mime_type=mime_type)


def load_face_pair(
    image_a: ImageInput, image_b: ImageInput
) -> tuple[str, Noul, tuple[ImageInput, ImageInput]]:
    """Build the face-pair state and question. Repo-only: needs a checkout.

    The evals builder needs an LFW pair record. This loader makes a placeholder
    record; its gold label is not used. Replace this function with your own
    state text and Noul in another project.

    Args:
        image_a: Image 1.
        image_b: Image 2.

    Returns:
        The state text, the same-person Noul and both images in order.

    Raises:
        TypeError: When the builder's same-person question is not a Noul.
    """
    pair = LfwPair(1, LfwFace("image_a", 1), LfwFace("image_b", 1), same_person=False)
    request = build_face_match_request(
        pair,
        left_image=image_a.data,
        right_image=image_b.data,
        mime_type=image_a.mime_type,
    )
    noul = request.questions[SAME_PERSON]
    if not isinstance(noul, Noul):
        raise TypeError(f"{SAME_PERSON} is not a Noul")
    return request.state, noul, (image_a, image_b)


def main() -> None:
    """Judge whether two face images show the same person and print P(yes)."""
    # 1. Read both images from the paths in the environment.
    image_a = read_image("TYPEVET_EXAMPLE_IMAGE_A")
    image_b = read_image("TYPEVET_EXAMPLE_IMAGE_B")
    state, noul, images = load_face_pair(image_a, image_b)
    print(f"state: {state}")

    # 2. Open the judgment session that TYPEVET_BACKEND selects.
    with open_judgment() as session:
        # 3. Ask the Noul about the state, conditioned on both images.
        response = session.port.judge(
            state, {"same_person": noul}, session.model, media=images
        )

    # 4. Print the probability of yes, then the winner last.
    p_yes = response.nouls["same_person"].noul
    print(f"P(yes): {p_yes:.4f}")
    print(f"winner: {'yes' if p_yes > THRESHOLD else 'no'}")


if __name__ == "__main__":
    main()

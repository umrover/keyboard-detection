import matplotlib.pyplot as plt
from ultralytics.engine.results import Boxes, Results

from keyrover.util import *
from keyrover.vision.bbox import *
from .image import img_to_numpy, ImageType, reorder_image_axes
from .math import get_median_factors


def normalize(img: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """
    Normalizes an array to the range [0, 1]
    """
    img -= img.min()
    img /= img.max()
    return img


def _imshow(img: ImageType, ax: plt.Axes | None) -> plt.Axes:
    """
    Plots an array using matplotlib
    """
    img = img_to_numpy(img)

    # if the image's data is out of bounds for matplotlib plotting:
    #   - any negative values
    #   - values > 255 and dtype is an integer
    #   - values > 1 and dtype is a float
    if img.min() < 0 or (img.max() > 255 and img.dtype.kind in {"i", "u"}) or (img.max() > 1 and img.dtype.kind == "f"):
        img = normalize(img.astype("float"))

    # matplotlib cannot plot bools, so we convert to uint8
    if img.dtype == bool:
        img = img.astype("uint8")

    # in some cases, the channel comes as the first axis, i.e. (c, w, h)
    # so we reorder to get (w, h, c)
    img = reorder_image_axes(img)

    if img.shape[-1] == 2:
        r, g = cv2.split(img)
        black = np.zeros(r.shape, dtype=img.dtype)
        img = cv2.merge([r, black, g])

    ax.axis("off")
    ax.imshow(img, interpolation="nearest")
    return ax


def imshow(img: ImageType,
           mask: ImageType | None = None,
           ax: plt.Axes | None = None,
           **kwargs) -> plt.Axes | tuple[plt.Axes, ...]:
    """
    Sets up and plots an image using matplotlib
    """

    # this is because when mask is specified, we create a 2-column figure
    # but when we specify, we use an existing subplot
    # hence, both cannot be specified
    if mask is not None and ax is not None:
        raise ValueError("Can't specify both mask and axis!")

    if mask is None:
        if ax is None:
            plt.figure()
            ax = plt.gca()
        return _imshow(img, ax)

    _, (ax1, ax2) = plt.subplots(ncols=2, **kwargs)
    _imshow(img, ax1)
    _imshow(mask, ax2)
    return ax1, ax2


def gridplot(n: int, **kwargs) -> tuple[plt.Figure, Sequence[plt.Axes]]:
    # creates a grid with the optimal dimensions (more square & few empty spots)
    # that will fit the images
    a, b = get_best_grid(n)
    fig, axes = plt.subplots(a, b, **kwargs)

    if a == b == 1:
        return fig, [axes]

    # flatten array
    axes = [ax for row in axes for ax in row]
    return fig, axes


def show_images(images: Sequence[ImageType],
                **kwargs) -> None:
    """
    Plots a grid of images using matplotlib
    """
    _, axes = gridplot(len(images), **kwargs)

    for i, ax in enumerate(axes):
        axes[i].axis("off")
        if i < len(images):
            imshow(images[i], ax=axes[i])

    plt.tight_layout()


def show_channels(img: ImageType) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Shows the RGB channels of an image
    """
    img = img_to_numpy(img)

    r, g, b = cv2.split(img)
    z = np.zeros(r.shape)

    show_images([img, np.stack([z, z, b]), np.stack([r, z, z]), np.stack([z, g, z])])
    return r, g, b


def imhist(img: ImageType | str,
           ax: plt.Axes | None = None,
           bins: int = 50,
           alpha: float = 0.5,
           histtype: Literal["bar", "barstacked", "step", "stepfilled"] = "stepfilled",
           **kwargs):
    """
    Plots a histogram of the RGB pixels in an image
    """
    if isinstance(img, str):
        img = cv2.imread(img)
    else:
        img = img_to_numpy(img)

    try:
        r, g, b = img.T
    except ValueError:
        r, g, b, _ = img.T

    if ax is None:
        ax = plt.gca()

    ax.axis("off")
    ax.hist(r.flatten(), color="#fa3c3c", bins=bins, alpha=alpha, histtype=histtype, **kwargs)
    ax.hist(g.flatten(), color="#74db95", bins=bins, alpha=alpha, histtype=histtype, **kwargs)
    ax.hist(b.flatten(), color="#42b3f5", bins=bins, alpha=alpha, histtype=histtype, **kwargs)


def get_best_grid(n: int) -> tuple[int, int]:
    """
    Returns the dimensions (a, b) needed to fit n squares.
    Minimizes empty squares (ab — n) and aspect ratio (a — b)
    """
    best_score = float("infinity")
    best_grid = None

    i = n
    while True:
        a, b = get_median_factors(i)
        diff = b - a
        empty = i - n

        if (score := diff + empty) < best_score:
            best_score = score
            best_grid = a, b

        if empty > best_score:
            break
        i += 1

    return best_grid


def draw_textbox(img: ImageType, bbox: LabeledBBox, color: Vec3 = (230, 55, 107), thickness: int = 3,
                 font=cv2.FONT_HERSHEY_SIMPLEX, font_size: float = 1.5, draw_text: bool = True) -> np.array:
    """
    Plots a box with text above it
    """

    img = img_to_numpy(img)
    cv2.rectangle(img, bbox.p1.astype("int"), bbox.p2.astype("int"), color=color, thickness=thickness)

    if draw_text:
        img = cv2.putText(img, bbox.label, bbox.p1.astype("int"),
                          font, font_size, (255, 255, 255), thickness, cv2.LINE_AA)
    return img


def plot_predictions(img: np.ndarray,
                     boxes: list[Boxes],
                     labels: Iterable,
                     scale: int = 4,
                     plot: bool = True,
                     draw_text: bool = True,
                     fig_kwargs: dict[str, Any] = {},
                     **kwargs) -> np.ndarray | None:
    """
    Plots a YOLO boxes object with labels
    """
    if fig_kwargs is None:
        fig_kwargs = {}
    size = img.shape
    img = cv2.resize(img, (scale * size[1], scale * size[0]))

    for cls, box in zip(labels, boxes):
        x1, y1, x2, y2 = map(lambda v: int(scale * v), box.xyxy[0])
        draw_textbox(img, (x1, y1), (x2, y2), str(cls), draw_text=draw_text, **kwargs)

    if plot:
        plt.figure(**fig_kwargs)
        imshow(img, ax=plt.gca())
    else:
        return img


def plot_yolo(results: Results, **kwargs) -> np.ndarray | None:
    return plot_predictions(results.orig_img, results.boxes,
                            [f"{box.cls} {box.conf}%" for box in results.boxes], **kwargs)


__all__ = ["imshow", "imhist", "show_images", "show_channels",
           "draw_textbox", "plot_yolo", "plot_predictions", "gridplot", "plt"]

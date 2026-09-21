"""Run the supplied ResNet-20 on the same CelebA protocol as ResNet-18."""

from reproduce_resnet18_celeba import main


if __name__ == "__main__":
    raise SystemExit(main(default_architecture="resnet20"))

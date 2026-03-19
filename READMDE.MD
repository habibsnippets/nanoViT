# nanoViT

## Theory behind ViT

The founding question of ViT: CNNs process images with sliding windows — they're inherently local. To see a long-range dependency (top-left corner relates to bottom-right), you need many stacked layers. Transformers see the entire sequence at once.

![step 1](https://i.postimg.cc/pdGmbYTc/Screenshot-from-2026-03-19-23-34-53.png)

Each patch becomes one token, exactly like each word becomes one token in GPT. The linear projection is learned — the network figures out what 256-dimensional "meaning" each patch should carry. After this step, the transformer doesn't know or care that the input was an image. It just sees a sequence of 64 vectors.


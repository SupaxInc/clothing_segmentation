import torch

LEARNING_RATE = 1e-3
MIN_LEARNING_RATE = 1e-06
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_EPOCHS = 50
NUM_WORKERS = 2
NUM_CLASSES = 7
IMAGE_HEIGHT = 384
IMAGE_WIDTH = 256
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "data/train/images/"
TRAIN_MASK_DIR = "data/train/masks/"
VAL_IMG_DIR = "data/validations/images/"
VAL_MASK_DIR = "data/validations/masks/"

# TODO: Convert this to a better data structure
CLASS_MAPPING = {
    0: [0],  # Background => [Background]
    1: [41],  # Skin => [Skin]
    2: [19], # Hair => [Hair]
    3: [4, 5, 6, 8, 11, 13, 14, 22, 24, 26, 35, 38, 46, 48, 49, 50, 51, 54, 55],  # Tops => [Blazer, Blouse, Bodysuit, Bra, Cardigan, Coat, Dress, Hoodie, Jacket, Jumper, Romper, Shirt, Suit, Sweater, Sweatshirt, Swimwear, T-shirt, Top, Vest]
    4: [25, 27, 30, 31, 40, 42, 53],  # Bottoms => [Jeans, Leggings, Panties, Pants, Shorts, Skirt, Tights]
    5: [7, 12, 16, 21, 28, 32, 36, 39, 43, 44, 58],  # Footwear => [Boots, Clogs, Flats, Heels, Loafers, Pumps, Sandals, Shoes, Sneakers, Socks, Wedges]
    6: [1, 2, 3, 9, 10, 15, 17, 18, 20, 23, 29, 33, 34, 37, 45, 47, 52, 56, 57],  # Accessories => [Accessories, Bag, Belt, Bracelet, Cape, Earrings, Glasses, Gloves, Hat, Intimate, Necklace, Purse, Ring, Scarf, Stockings, Sunglasses, Tie, Wallet, Watch]
}
CLASS_MAPPING_NAMING = {
    0: "Background",
    1: "Skin",
    2: "Hair",
    3: "Tops",
    4: "Bottoms",
    5: "Footwear",
    6: "Accessories"
}
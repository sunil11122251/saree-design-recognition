import os
from PIL import Image, ImageDraw

dataset_path = r"C:\Users\Hi\saree-design-recognition\data\sarees_dataset\handloom_sarees"

output_path = r"C:\Users\Hi\saree-design-recognition\results\deeplure_contact_sheet.jpg"

image_files = []

for file in os.listdir(dataset_path):
    if file.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        image_files.append(file)

image_files.sort()

# Number of images to display
max_images = min(50, len(image_files))

# Thumbnail settings
thumb_width = 180
thumb_height = 180
label_height = 25
columns = 5
rows = (max_images + columns - 1) // columns

sheet_width = columns * thumb_width
sheet_height = rows * (thumb_height + label_height)

sheet = Image.new("RGB", (sheet_width, sheet_height), "white")
draw = ImageDraw.Draw(sheet)

for i in range(max_images):

    image_path = os.path.join(dataset_path, image_files[i])

    try:
        image = Image.open(image_path).convert("RGB")
        image.thumbnail((thumb_width - 10, thumb_height - 10))

        x = (i % columns) * thumb_width
        y = (i // columns) * (thumb_height + label_height)

        # Center image
        image_x = x + (thumb_width - image.width) // 2
        image_y = y + (thumb_height - image.height) // 2

        sheet.paste(image, (image_x, image_y))

        draw.text(
            (x + 5, y + thumb_height),
            image_files[i][:20],
            fill="black"
        )

    except Exception as e:
        print("Could not process:", image_files[i], e)

sheet.save(output_path)

print("Contact sheet created:")
print(output_path)
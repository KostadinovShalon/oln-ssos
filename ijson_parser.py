import json

import ijson


file = '/home/brian/Documents/datasets/openimages/detection_train.json'
def load_images(json_filename):
    counter = 0
    images = []
    annotations = []
    categories = []
    with open(json_filename, 'rb') as input_file:
        # load json iteratively
        parser = ijson.parse(input_file)
        loading_images = False
        loading_annotations = False
        loading_categories = False
        current_img = {
            "id": None,
            "width": None,
            "height": None,
            "file_name": None
        }
        current_ann = {
            "id": None,
            "image_id": None,
            "area": None,
            "iscrowd": None,
            "category_id": None,
            "bbox": list()
        }
        current_category = {
            "id": None,
            "name": None,
        }
        for prefix, event, value in parser:
            if event == 'start_array' and prefix == 'images':
                loading_images = True
                continue
            if event == 'start_array' and prefix == 'annotations':
                loading_annotations = True
                continue
            if event == 'start_array' and prefix == 'categories':
                loading_categories = True
                continue
            if loading_images:
                if counter > 100:
                    counter = 0
                    loading_images = False
                all_are_not_none = all(v is not None for v in current_img.values())
                if all_are_not_none:
                    images.append(current_img)
                    print('Parsed images={}'.format(len(images)))
                    current_img = {
                        "id": None,
                        "width": None,
                        "height": None,
                        "file_name": None
                    }
                    counter += 1
                else:
                    if event == 'number' and prefix == 'images.item.id':
                        current_img['id'] = int(value)
                    elif event == 'number' and prefix == 'images.item.width':
                        current_img['width'] = int(value)
                    elif event == 'number' and prefix == 'images.item.height':
                        current_img['height'] = int(value)
                    elif event == 'string' and prefix == 'images.item.file_name':
                        first_letter = value[0]
                        current_img['file_name'] = f'train_{first_letter}/{value}'
                if prefix == 'images' and event == 'end_array':
                    loading_images = False

            if loading_annotations:
                ready = current_ann['id'] is not None and \
                        current_ann['image_id'] is not None and \
                        current_ann['area'] is not None and \
                        current_ann['iscrowd'] is not None and \
                        current_ann['category_id'] is not None and \
                        len(current_ann['bbox']) == 4
                if ready:
                    annotations.append(current_ann)
                    print('Parsed annotations={}'.format(len(annotations)))
                    current_ann = {
                        "id": None,
                        "image_id": None,
                        "area": None,
                        "iscrowd": None,
                        "category_id": None,
                        "bbox": list()
                    }
                    counter += 1
                else:
                    if event == 'number' and prefix == 'annotations.item.id':
                        current_ann['id'] = int(value)
                    elif event == 'number' and prefix == 'annotations.item.image_id':
                        current_ann['image_id'] = int(value)
                    elif event == 'number' and prefix == 'annotations.item.area':
                        current_ann['area'] = int(value)
                    elif event == 'number' and prefix == 'annotations.item.iscrowd':
                        current_ann['iscrowd'] = int(value)
                    elif event == 'number' and prefix == 'annotations.item.category_id':
                        current_ann['category_id'] = int(value)
                    elif event == 'number' and prefix == 'annotations.item.bbox.item':
                        current_ann['bbox'].append(int(value))
                if prefix == 'annotations' and event == 'end_array':
                    loading_annotations = False
                if counter > 100:
                    break
            if loading_categories:
                ready = current_category['id'] is not None and \
                        current_category['name'] is not None
                if ready:
                    categories.append(current_category)
                    print('Parsed categories={}'.format(len(categories)))
                    current_category = {
                        "id": None,
                        "name": None
                    }
                else:
                    if event == 'number' and prefix == 'categories.item.id':
                        current_category['id'] = int(value)
                    elif event == 'string' and prefix == 'categories.item.name':
                        current_category['name'] = value
                if prefix == 'categories' and event == 'end_array':
                    loading_categories = False

    return images, annotations, categories

i, a, c = load_images(file)

data = {
    'info': {
        'year': 2022,
        'version': 7,
        'description': 'OpenImages datasets',
        "url": "https://storage.googleapis.com/openimages/web/index.html",
        "date_created": "10-10-2023"
    },
    'images': i,
    'annotations': a,
    'categories': c
}
print("Saving Json")
json.dump(data, open('/home/brian/Documents/datasets/openimages/detection_train_curated.json', 'w'))

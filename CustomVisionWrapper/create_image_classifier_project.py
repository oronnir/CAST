""" Create a new Image Classification project

two folder, one train, one test,
each folder have sub-folder grouped by tag

this format only support multi-class, for multi-label

"""
import glob
import os

from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateBatch
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateEntry
from msrest.authentication import ApiKeyCredentials

from config import training_key, ENDPOINT

MAX_IMAGES = 40


class ICProjectCreator(object):
    def __init__(self):
        self.credentials = ApiKeyCredentials(in_headers={"Training-key": training_key})
        trainer = CustomVisionTrainingClient(ENDPOINT, self.credentials)
        self.training_api = trainer

    def create_project(self, training_path, dataset_name):
        # Create a new project
        print("Creating project...")
        project_name = dataset_name
        project = self.training_api.create_project(project_name)
        print("Project Id: {}".format(project.id))

        if os.path.exists(training_path):
            all_classes = [c for c in glob.glob(os.path.join(training_path, '*'))]
            print("add images for dataset {0}".format(dataset_name))
            for class_path in all_classes:
                img_paths = glob.glob(os.path.join(class_path, '*'))
                tag_name = class_path.split('\\')[-1]
                idx_tag = self.training_api.create_tag(project.id, tag_name)
                if img_paths:
                    print("add images for category {0} from {1}".format(idx_tag.name, class_path))
                    list_of_images = []
                    list_of_tags = []
                    image_count = 0
                    for img_path in img_paths:
                        with open(img_path, mode="rb") as img_data:
                            list_of_images.append(ImageFileCreateEntry(name=img_path, contents=img_data.read(),
                                                                       tag_ids=[idx_tag.id]))
                            list_of_tags.append(idx_tag.id)

                            # Updates the image count the upload the batch if the count reached MAX_IMAGES
                            image_count += 1
                            if image_count == MAX_IMAGES:
                                upload_result = self.training_api\
                                    .create_images_from_files(project.id, ImageFileCreateBatch(images=list_of_images))

                                if not upload_result.is_batch_successful:
                                    print("Image batch upload failed.")
                                    for image in upload_result.images:
                                        print("Image status: ", image.status)

                                list_of_images = []
                                list_of_tags = []
                                image_count = 0
                    if len(img_paths) % MAX_IMAGES != 0:
                        self.training_api\
                            .create_images_from_files(project.id, ImageFileCreateBatch(images=list_of_images))
            print("upload succeed for dataset {0}".format(dataset_name))
            return project
        else:
            print("file path {0} not exist".format(training_path))


def main():
    training_root = r'???\Labels\Training\Triplets'
    series = os.listdir(training_root)
    for ser in series:
        print('start creating a project for series: {}'.format(ser))
        ser_path = os.path.join(training_root, '', ser)
        dataset_name = 'Triplets_{}'.format(ser)
        creator = ICProjectCreator()
        creator.create_project(ser_path, dataset_name)
        print('Done creating a project for series {}'.format(ser))
    print('Done Done!')


if __name__ == '__main__':
    main()

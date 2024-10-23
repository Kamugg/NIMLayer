from enum import Enum
from PIL import Image
from openai import OpenAI
from pathlib import Path
import requests
from io import BytesIO


class DALLEAgent(object):
    class RESOLUTIONS(Enum):
        TINY = '256x256'
        SMALL = '512x512'
        LARGE = '1024x1024'
        TALL = '1024x1792'
        WIDE = '1792x1024'

    class MODELS(Enum):
        DALLE2 = 'dall-e-2'
        DALLE3 = 'dall-e-3'

    class QUALITIES(Enum):
        STANDARD = 'standard'
        HD = 'hd'

    class STYLES(Enum):
        VIVID = 'vivid'
        NATURAL = 'natural'

    class RETURN_MODE(Enum):
        URL = 'url'
        IMAGE = 'image'

    def __init__(self, key: str | Path):
        """
        Initializes the agent.
        :param key: The key for the API. Can be provided either as a raw string or as a path to the file containing it.
        """
        key_path = Path(key)
        # Retrieves key either by storing it in a variable or retrieving it from local file.
        if not key_path.is_file():
            loaded_key = key
        else:
            with open(key_path, 'r') as f:
                loaded_key = f.read()
                f.close()
        self.client = OpenAI(api_key=loaded_key)

    def __validate_str_enum(self, param: str | Enum, en: Enum, error_string: str) -> str:
        sanitized = None
        if isinstance(param, str):
            for it in en:
                if it.value == param:
                    sanitized = param
        elif isinstance(param, en):
            for it in en:
                if it == param:
                    sanitized = param.value
        if sanitized is None:
            raise ValueError(error_string)
        return sanitized

    def __sanitize_parameters(self, parameters: dict) -> dict:
        sanitized = {}

        # Check valid prompt

        if 'prompt' in parameters:
            if not isinstance(parameters['prompt'], str):
                raise ValueError(
                    f'Provided prompt must be a valid string and not {type(parameters["prompt"]).__name__}.')
            sanitized['prompt'] = parameters['prompt']

        # Check valid image

        if 'image' in parameters:
            if not isinstance(parameters['image'], (str, Path)):
                raise ValueError(
                    f'Provided image path must be either string or Path. Provided: {type(parameters["name"]).__name__}')
            if isinstance(parameters['image'], str):
                image_path = Path(parameters['image'])
            else:
                image_path = parameters['image']
            if not image_path.is_file():
                raise FileNotFoundError(f'Image not found at path {image_path}')
            if not image_path.suffix == '.png':
                raise ValueError(f'Invalid file format: provided image must be PNG, but {image_path.suffix} was given.')
            img = Image.open(image_path)
            if img.size[0] != img.size[1]:
                raise ValueError(f'Provided image must be a square. Got a {img.size[0]}x{img.size[1]} image.')
            sanitized['image'] = open(image_path, 'rb')

        # Check valid model id

        if 'model' in parameters:
            sanitized['model'] = self.__validate_str_enum(parameters['model'],
                                                          self.MODELS,
                                                          f'Provided model must be either \'dall-e-2\', \'dall-e-3\' or DALLEAgent.MODELS. Provided {parameters["model"]} of type {type(parameters["model"]).__name__}.')

        # Check numbers of images to generate

        if (not isinstance(parameters['n'], int)) or not (0 < parameters['n'] <= 10):
            raise ValueError(f'Provided n must be a valid integer in the range 1 to 10. Provided: {parameters["n"]}')
        sanitized['n'] = parameters['n']

        # Check provided quality id

        if 'quality' in parameters:
            sanitized['quality'] = self.__validate_str_enum(parameters['quality'],
                                                            self.QUALITIES,
                                                            f'Provided quality must be either a valid string  \'standard\' or \'hd\' or DALLEAgent.QUALITIES. Provided {parameters["quality"]} of type {type(parameters["quality"]).__name__}.')

        # Check valid resolution

        sanitized['size'] = self.__validate_str_enum(parameters['size'],
                                                     self.RESOLUTIONS,
                                                     'Provided resolution must be either a valid string or DALLEAgent.RESOLUTIONS. Provided resolution is {parameters["size"]} of type {type(parameters["size"]).__name__}.')

        # Check valid style

        if 'style' in parameters:
            if parameters['style'] is not None:
                sanitized['style'] = self.__validate_str_enum(parameters['style'],
                                                              self.STYLES,
                                                              f'Provided style must be either "natural", "vivid" or DALLEAgent.STYLES. Provided style is {parameters["size"]} of type {type(parameters["size"]).__name__}.')

        # Logic checks since dalle3 and dalle2 have some restrictions

        if sanitized['model'] == self.MODELS.DALLE2.value:
            if sanitized.get('quality', None) is not None and sanitized['quality'] == self.QUALITIES.HD.value:
                raise ValueError('HD quality is only supported by dall-e 3.')
            if sanitized['size'] not in ['256x256', '512x512', '1024x1024']:
                raise ValueError(
                    f'Provided size is not supported by dall-e 2. Must be either 256x256, 512x512 or 1024x1024')
            if 'style' in sanitized:
                raise ValueError(f'Style parameter is not supported by dall-e 2.')
        else:
            if sanitized['n'] > 1:
                raise ValueError(f'Invalid value for n: {sanitized["n"]}. dall-e 3 only supports n=1.')

            if sanitized['size'] not in ['1024x1024', '1792x1024', '1024x1792']:
                raise ValueError(
                    f'Provided size is not supported by dall-e 3. Must be either 1024x1024, 1792x1024 or 1024x1792')
        return sanitized

    def __fetch_images(self, data: list):
        images = []
        for image in data:
            response = requests.get(image.url)
            img_data = BytesIO(response.content)
            images.append(Image.open(img_data))
        return images

    def generate_from_prompt(self,
                             prompt: str,
                             model: MODELS | str = MODELS.DALLE2,
                             n: int = 1,
                             quality: QUALITIES | str = QUALITIES.STANDARD,
                             size: RESOLUTIONS | str = RESOLUTIONS.TINY,
                             style: STYLES | str | None = None,
                             return_mode: RETURN_MODE | str = RETURN_MODE.IMAGE
                             ):
        # Check valid return type

        return_mode = self.__validate_str_enum(return_mode,
                                               self.RETURN_MODE,
                                               f'Provided return mode must be either "url" or "image" or DALLEAgent.RETURN_MODE. Provided mode is {return_mode} of type {type(return_mode).__name__}.')
        to_sanitize = {'prompt': prompt,
                       'model': model,
                       'n': n,
                       'quality': quality,
                       'size': size,
                       'style': style}
        sanitized = self.__sanitize_parameters(to_sanitize)
        response = self.client.images.generate(**sanitized)

        if return_mode == 'url':
            urls = [img.url for img in response.data]
            return urls
        else:
            return self.__fetch_images(response.data)

        return response

    def image_variation(self,
                        image: str | Path,
                        n: int = 1,
                        size: RESOLUTIONS | str = RESOLUTIONS.TINY,
                        return_mode: RETURN_MODE | str = RETURN_MODE.IMAGE
                        ):
        # Check valid return type

        return_mode = self.__validate_str_enum(return_mode,
                                               self.RETURN_MODE,
                                               f'Provided return mode must be either "url" or "image" or DALLEAgent.RETURN_MODE. Provided mode is {return_mode} of type {type(return_mode).__name__}.')
        to_sanitize = {'image': image,
                       'model': self.MODELS.DALLE2,
                       'n': n,
                       'size': size
                       }
        sanitized = self.__sanitize_parameters(to_sanitize)
        response = self.client.images.create_variation(**sanitized)
        if return_mode == 'url':
            urls = [img.url for img in response.data]
            return urls
        else:
            return self.__fetch_images(response.data)

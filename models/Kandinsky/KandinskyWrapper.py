from diffusers import Kandinsky3Img2ImgPipeline
from pathlib import Path
import PIL
import torch
import os


class KandinskyWrapper(object):
    """
    Wrapper class for the Kandinsky model.
      A wrapper around the Kandinsky3 Img2Img pipeline for generating images..

      Attributes:
          pipeline (Kandinsky3Img2ImgPipeline): The diffusion pipeline for image generation.
          base_prompt (str): A default prompt to guide the generation process.
          base_negative_prompt (str): A default negative prompt to avoid undesirable outputs.
    """

    def __init__(self, model_path: str | Path | None = None):

        """
        Initializes the KandinskyWrapper and sets up the diffusion pipeline.

        Args:
            model_path (str | Path | None): Path to the pre-trained model directory. Defaults to the "kandinsky-community/kandinsky-3" checkpoint and downloads it.

        Raises:
            ValueError: If the model path is invalid or not accessible.
        """

        if model_path is None:
            model_path = "kandinsky-community/kandinsky-3"
        self.pipeline = Kandinsky3Img2ImgPipeline.from_pretrained(model_path, variant="fp16", torch_dtype=torch.float16,
                                                                  use_safetensors=True)
        if model_path == "kandinsky-community/kandinsky-3":
            self.pipeline.save_pretrained('kandinsky_checkpoints', variant="fp16", torch_dtype=torch.float16,
                                          use_safetensors=True)
        self.pipeline.to(device='cuda')
        self.pipeline.enable_model_cpu_offload()
        self.pipeline.set_progress_bar_config(leave=False, disable=True)
        self.base_prompt = '''
    A vibrant anime-style illustration of the input image, featuring clean line art, 
    bright colors, and smooth shading. The scene should have an expressive and dynamic aesthetic typical of anime, 
    with fine attention to detail in facial features, hair highlights, and environmental elements. 
    Maintain the essence of the original image while enhancing it with the charm and stylization of Japanese animation. 
    The style should resemble that of a high-quality anime studio production, with a whimsical and visually appealing design.
    '''
        self.base_negative_prompt = 'blurry, distorted, extra limbs, watermark, text, bad anatomy, low quality,'

    def __get_random_seed(self) -> int:

        """
        Generate a random seed using os.urandom.

        Returns:
            int: A random integer seed for reproducibility in image generation.
        """

        RAND_SIZE = 8
        random_data = os.urandom(RAND_SIZE)
        random_seed = int.from_bytes(random_data, byteorder="big")
        return random_seed

    def __preprocess(self, params: dict) -> dict:

        """
        Preprocesses input parameters for the pipeline.

        Args:
            params (dict): A dictionary containing input parameters such as:
                - prompt (str | list[str]): The generation prompt(s).
                - image (str | Path | PIL.Image.Image): The input image.
                - negative_prompt (str | list[str]): Negative prompt(s).
                - num_inference_steps (int): Number of inference steps.
                - strength (float): Image strength for generation.
                - guidance_scale (float): Guidance scale for prompt adherence.
                - seed (int | None): Seed for reproducibility.

        Returns:
            dict: A preprocessed dictionary of parameters ready for the pipeline.

        Raises:
            ValueError: If parameters are of incorrect types or values.
        """

        # Check if prompts are a string or a list of strings

        prompts = []
        if isinstance(params['prompt'], str):
            prompts = [self.base_prompt + params['prompt']]
        elif isinstance(params['prompt'], list):
            for p in params['prompt']:
                if not isinstance(p, str):
                    raise ValueError('prompt must be a string or a list of strings')
                else:
                    prompts.append(self.base_prompt + p)
        else:
            raise ValueError('prompt must be a string or a list of strings')

        params['prompt'] = prompts

        # Validate and resize image:

        if not isinstance(params['image'], PIL.Image.Image):
            if isinstance(params['image'], str):
                params['image'] = Path(params['image'])

            if not params['image'].is_file():
                raise ValueError('image must be a valid file path')

            params['image'] = PIL.Image.open(params['image'])

        # Convert image sizes to multiples of 64, otherwise pipeline throws an exception
        # These are the closest multiples to avoid image quality degradation

        w, h = params['image'].size
        w = round(w / 64) * 64
        h = round(h / 64) * 64
        params['image'] = params['image'].resize((w, h))

        nprompts = []
        if isinstance(params['negative_prompt'], str):
            nprompts = [self.base_negative_prompt + params['negative_prompt']]
        elif isinstance(params['negative_prompt'], list):
            for p in params['negative_prompt']:
                if not isinstance(p, str):
                    raise ValueError('negative_prompt must be a string or a list of strings')
                else:
                    nprompts.append(self.base_negative_prompt + p)
        else:
            raise ValueError('negative_prompt must be a string or a list of strings')
        params['negative_prompt'] = nprompts

        if len(params['prompt']) != len(nprompts):
            params['negative_prompt'] = [nprompts[0]] * len(params['prompt'])

        if not isinstance(params['num_inference_steps'], int):
            raise ValueError('num_inference_steps must be an integer')

        if params['num_inference_steps'] < 1:
            params['num_inference_steps'] = 20

        if not isinstance(params['strength'], float):
            raise ValueError('Strength must be a floating point number')

        if params['strength'] < 0 or params['strength'] > 1:
            params['strength'] = 0.4

        if not isinstance(params['guidance_scale'], float):
            raise ValueError('Strength must be a floating point number')

        if params['guidance_scale'] < 0:
            params['guidance_scale'] = 4.0

        if params['seed'] is None:
            params['seed'] = self.__get_random_seed()
        elif not isinstance(params['seed'], int):
            raise ValueError('seed must be an integer')

        params['generator'] = torch.Generator().manual_seed(params['seed'])

        return params

    def __call__(self,
                 prompt: str | list[str],
                 image: str | Path | PIL.Image.Image,
                 negative_prompt: str | list[str] = '',
                 num_inference_steps: int = 100,
                 strength: float = 0.7,
                 guidance_scale: float = 6.0,
                 seed: int | None = None):

        """
        Runs the image generation process using the provided parameters.

        Args:
            prompt (str | list[str]): The generation prompt(s).
            image (str | Path | PIL.Image.Image): The input image or its path.
            negative_prompt (str | list[str], optional): Negative prompt(s) for undesirable features. Defaults to ''.
            num_inference_steps (int, optional): Number of steps for the diffusion process. Defaults to 100.
            strength (float, optional): Strength for modifying the input image. Defaults to 0.7.
            guidance_scale (float, optional): Scale for adhering to the prompt. Defaults to 6.0.
            seed (int | None, optional): Seed for reproducibility. Defaults to None.

        Returns:
            dict: A dictionary containing generated images and metadata:
                - 'images': List of generated images with prompts.
                - 'seed': Seed used for generation.

        Raises:
            ValueError: If parameters are invalid or preprocessing fails.
        """

        params = {
            'prompt': prompt,
            'image': image,
            'negative_prompt': negative_prompt,
            'num_inference_steps': num_inference_steps,
            'strength': strength,
            'guidance_scale': guidance_scale,
            'seed': seed
        }
        params = self.__preprocess(params)
        raw_output = self.pipeline(**{k: v for k, v in params.items() if k != 'seed'})
        output = {'images': [], 'seed': params['seed']}
        for i, img in enumerate(raw_output.images):
            img_data = {'image': img}
            img_data['prompt'] = params['prompt'][i]
            img_data['negative_prompt'] = params['negative_prompt'][i]
            output['images'].append(img_data)
        return output

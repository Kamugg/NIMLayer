# KolorsWrapper

# How to use

To initialize the wrapper just create an instance of it:

```wrapper = KolorsWrapper()```

The only parameter required is the checkpoint location. If none is provided the underlying diffusion model will be downloaded
from scratch and then saved in the ```kolors_diffusers``` directory.

## The ```__call__``` method

### Input

To create an image simply call the ```__call__``` method on the desired prompt:

```wrapper('Image of a turtle')```

Will return an image of a turtle like this

![](readme_images/turtle_img.png)

The list of arguments supported by the ```__call__``` method is as follows:

- **prompt**: The provided prompt either in the form of a string or a list of strings if you want to provide more than one.
- **seed**: Set a seed manually for te diffusion model. If none is provided the wrapper samples one randomly.
- **return_pt**: Boolean that if True makes the diffuser return PyTorch tensors instead of PIL images. Defaults to False.
- **num_inference_steps**: Integer that sets how many denoising steps the model does to generate an image. Higher values may correlate
with higher quality images, but it's not always the case. This parameter defaults to 25. Here's one example of an images 
generated:

50 inference steps
![50 inference steps](readme_images/inf50.png)

2 inference steps
![2 inference steps](readme_images/inf2.png)

- **conditioning**: A string that is appended to every prompt provided. Useful if you are passing many different prompts
and you want all of them to begin with 'A painting of...'. Defaults to None.
- **images_per_prompt**: Integer that sets how many images to generate for any given prompt. Defaults to 1.
- **guidance_scale**: Float value that sets "how closely" the diffuser needs to follow the provided prompt. Defaults to 5.0.
Setting the guidance scale too high can end up in corrupted images, like in this case where I set it to 50:

![](readme_images/highguid.png)

On the other hand a value too low can end up with "malformed" images, like in this example where the guidance scale is 1:

![2 inference steps](readme_images/lowguid.png)

### Output

The wrapper returns a dictionary with the following fields:

- **seed**: The seed used for the generation
- **images**: The list of generated images, the length of the list is N*IPP where N is the number of provided prompts and IPP
is the ```images_per_prompt``` parameter value.
- **guidance_scale**: Value of the parameter ```guidance_scale``` used for the generation.

## Requirements

**diffusers**==0.30.3

**xformers**==0.0.28.post1

**triton**==3.0.0

**torchvision**==0.19.1

**torch**==2.4.1
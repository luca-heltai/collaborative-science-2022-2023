To create a image from a Docker file, create a directory, and a empty Dockerfile
in it

    docker/heltai/Docker

the command

    docker build -t heltai/example-image .

in the directory `docker/heltai/` generates an image containing the result of
the commands in the Dockerfile.

To mount a local directory when running an image do:

    docker run -t -i -v /local/directory/path:/path/in/container image/torun

this will run the image interactively (`-i`), using a termina (`-t`) and
mounting (`-v`) the directory `/local/directory/path` in `/path/in/container`.
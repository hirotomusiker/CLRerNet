
# Installation Tips

## Docker-Compose

If you do not have a docker-compose environment:
```
sudo apt install docker-compose
sudo service docker start
```

To install NMS extension, nvidia runtine must be enabled during build.
Add `default-runtime` to `/etc/docker/daemon.json` as follows:  
```
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "default-runtime": "nvidia"
}
```

```bash
sudo systemctl restart docker
docker-compose build --build-arg UID="`id -u`" dev
docker-compose run --rm dev
```

In case NMS installation still fails, skip `RUN python /tmp/nms/setup.py install` in `Dockerfile` and build `nms` inside the container:
```
cd libs/models/layers/nms/
python setup.py install
cd /work
```
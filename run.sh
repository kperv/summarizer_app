#!/usr/bin/env bash

docker run -it -p 8080:5000 -v $(pwd):/sum_app kperv/sum_app:0.2 /sum_app/app.py


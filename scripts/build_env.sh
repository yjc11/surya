run_container() {
  echo "run docker image"
  DOCKER_IMAGE="112a05617efe"
  DEV_MOUNT="-v /workspace:/workspace"
  container_name="surya_dev_yjc"
  docker run --pid=host --name ${container_name} ${DEV_MOUNT} -p 9199:9199 --shm-size=20g --ulimit memlock=-1 -d -it --ipc=host --gpus 'all' $DOCKER_IMAGE bash
  if [ $? -eq 0 ]; then
      echo "run  docker image success"
  else
      echo "run docker image failed!"
      exit 1
  fi
}

run_container
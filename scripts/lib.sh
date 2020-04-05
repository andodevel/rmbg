#!/usr/bin/env bash

function resolve_script_source {
    SOURCE=${BASH_SOURCE[0]}
    while [ -h "$SOURCE" ]; do
      DIR=$( cd -P $( dirname "$SOURCE") && pwd )
      SOURCE=$(readlink "$SOURCE")
      [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
    done
    DIR=$( cd -P $( dirname "$SOURCE" ) && pwd )
    echo $DIR
}




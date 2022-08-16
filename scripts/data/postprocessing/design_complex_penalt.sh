#!/bin/bash

# compute what directory this file is in
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
# DIR is the directory this file is in, e.g. postprocessing
DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"

cd $DIR
source ../../config.sh

echo 'My input etab: ' $1
echo 'My output : ' $2
echo $1 > $2
$MST_PATH/bin/enerTable --e $1 --opt 2000000 --kTi 10 --kTf 0.1 --cyc 10 --lc 1 --randomSeed>> $2

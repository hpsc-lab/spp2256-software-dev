#!/bin/bash

set -euo pipefail

PORT=${1-1729}

# We need to create a reverse tunnel from the
# compute node to the login node.
if [[ "${HOSTNAME}" == "licca047" ]]; then
    function cleanup() {
        echo "Killing process with ID ${PROXY_PID}..."
        kill "${PROXY_PID}"
    }

   REMOTE=licca020

   # Forward the port back to the login node.
   ssh -N -R "${PORT}:localhost:${PORT}" "${REMOTE}" &
   # Remember the PID
   PROXY_PID=$!
   # At the end of this script, terminate the SSH forwarding.
   trap cleanup EXIT
fi

if [[ "${HOSTNAME}" == "licca047" ]]; then
    SERVER="licca-li-02.rz.uni-augsburg.de"
elif [[ "${HOSTNAME}" == *".rc.ucl.ac.uk" ]]; then
    # Fibonacci or Mandelbrot
    SERVER="${HOSTNAME}"
    export JULIAUP_CHANNEL=1.11 # Make sure we use the right channel
    export JULIA_NUM_THREADS=16
fi

echo "On your local machine run
    ssh -f -N ${SERVER} -L ${PORT}:localhost:${PORT}
"

julia --project -e "
import Pkg
Pkg.instantiate()
import Pluto
Pluto.run(; launch_browser=false, port=${PORT}, auto_reload_from_file=true)
"

#!/bin/bash

# Get all config files
start_directory="configs"
configs=()
while IFS= read -r -d $'\0' file; do
    configs+=("$file")
done < <(find "$start_directory" -type f -print0)


# Select a file
config=""
PS3="Select a config please: "
while [ -z $config ]; do
    select selected_config in "${configs[@]}" Quit; do
        if [ $REPLY -eq $((${#configs[@]}+1)) ]
        then
            echo "Bye!"
            exit 0
        elif [ $REPLY -gt $((${#configs[@]})) ] || [ $REPLY -lt 1 ]
        then
            echo "Unlisted config, please try again"
            continue
        fi
        
        config=$selected_config
        break
    done
done

mode=""
PS3="Select a mode please: "
while [ -z $mode ]; do
    select selected_mode in "Run" "Run w/ Bindings (Dev Mode)" "Compile" "Compile & Run" Quit; do
        if [ $REPLY -eq 5 ]
        then
            echo "Bye!"
            exit 0
        elif [ $REPLY -gt 5 ] || [ $REPLY -lt 1 ]
        then
            echo "Unlisted mode, please try again"
            continue
        fi
        
        mode=$REPLY
        break
    done
done

case $mode in

  1)
    echo "Running: \"singularity run --writable-tmpfs DIALS.sif python experiment.py with ./${config}\""
    singularity run --writable-tmpfs DIALS.sif python experiment.py with ./${config}
    ;;

  2)
    echo "Running: \"singularity run --writable-tmpfs --bind ./simulators:/simulators/ DIALS.sif python experiment.py with ./${config}\""
    singularity run --writable-tmpfs --bind ./simulators:/simulators DIALS.sif python experiment.py with ./${config}
    ;;

  3)
    echo "Compiling (This will take a while)"
    echo "Running: \"sudo singularity build DIALS.sif DIALS.def\""
    sudo singularity build DIALS.sif DIALS.def
    echo "Done!!"
    ;;

  4)
    echo "Compiling (This will take a while)"
    echo "Running: \"sudo singularity build DIALS.sif DIALS.def\""
    sudo singularity build DIALS.sif DIALS.def
    echo "Running!"
    echo "Running: \"singularity run --writable-tmpfs DIALS.sif python experiment.py with ./${config}\""
    singularity run --writable-tmpfs DIALS.sif python experiment.py with ./${config}
    ;;
esac


echo $mode

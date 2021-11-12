#!/bin/bash
sh -c "cd ~/codeden/torch-hd/ && echo "${PWD}" && pip uninstall -y torch_hd && python setup.py install"

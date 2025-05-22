rm -f watercolor.zip*

gdown 1fa2L6oaPSjZ1_WqlTmIp6i2RbdR2y1Pw

names=(clipart watercolor)
for name in "${names[@]}"
do
    unzip ${name}.zip
    rm ${name}.zip
done
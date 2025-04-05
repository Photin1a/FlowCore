#!/bin/bash
rm compile_commands.json
touch compile_commands.json

cat ./build/*/compile_commands.json >> compile_commands.json
sed -i 's/]\[/,/g' compile_commands.json
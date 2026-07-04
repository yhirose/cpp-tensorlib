# Portable bin2c: turn a binary file into a comma-separated C byte list that
# cuda.h #includes as an array initializer (the off-Apple compilers predate
# C23 #embed). No xxd/python dependency — CMake's file(READ ... HEX) only.
# Appends a 0x00 terminator, which cuModuleLoadData requires for a PTX image.
#   cmake -DINPUT=x.ptx -DOUTPUT=x.inc -P cmake/bin2c.cmake
file(READ "${INPUT}" hex HEX)
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," bytes "${hex}")
file(WRITE "${OUTPUT}" "${bytes} 0x00\n")

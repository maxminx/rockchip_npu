# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.23

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/sun/myprojects/mygithub/rockchip_npu/telpo_AlgSDK_rk1109_intelif

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/sun/myprojects/mygithub/rockchip_npu/telpo_AlgSDK_rk1109_intelif/build

# Include any dependencies generated for this target.
include CMakeFiles/telpoalgsdk.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/telpoalgsdk.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/telpoalgsdk.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/telpoalgsdk.dir/flags.make

CMakeFiles/telpoalgsdk.dir/src/telpo_algsdk.cpp.o: CMakeFiles/telpoalgsdk.dir/flags.make
CMakeFiles/telpoalgsdk.dir/src/telpo_algsdk.cpp.o: ../src/telpo_algsdk.cpp
CMakeFiles/telpoalgsdk.dir/src/telpo_algsdk.cpp.o: CMakeFiles/telpoalgsdk.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sun/myprojects/mygithub/rockchip_npu/telpo_AlgSDK_rk1109_intelif/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/telpoalgsdk.dir/src/telpo_algsdk.cpp.o"
	/home/sun/Desktop/mydir/2-projects/arm/V61_sdk_out/prebuilts/gcc/linux-x86/arm/gcc-linaro-6.3.1-2017.05-x86_64_arm-linux-gnueabihf/bin/arm-linux-gnueabihf-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/telpoalgsdk.dir/src/telpo_algsdk.cpp.o -MF CMakeFiles/telpoalgsdk.dir/src/telpo_algsdk.cpp.o.d -o CMakeFiles/telpoalgsdk.dir/src/telpo_algsdk.cpp.o -c /home/sun/myprojects/mygithub/rockchip_npu/telpo_AlgSDK_rk1109_intelif/src/telpo_algsdk.cpp

CMakeFiles/telpoalgsdk.dir/src/telpo_algsdk.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/telpoalgsdk.dir/src/telpo_algsdk.cpp.i"
	/home/sun/Desktop/mydir/2-projects/arm/V61_sdk_out/prebuilts/gcc/linux-x86/arm/gcc-linaro-6.3.1-2017.05-x86_64_arm-linux-gnueabihf/bin/arm-linux-gnueabihf-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sun/myprojects/mygithub/rockchip_npu/telpo_AlgSDK_rk1109_intelif/src/telpo_algsdk.cpp > CMakeFiles/telpoalgsdk.dir/src/telpo_algsdk.cpp.i

CMakeFiles/telpoalgsdk.dir/src/telpo_algsdk.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/telpoalgsdk.dir/src/telpo_algsdk.cpp.s"
	/home/sun/Desktop/mydir/2-projects/arm/V61_sdk_out/prebuilts/gcc/linux-x86/arm/gcc-linaro-6.3.1-2017.05-x86_64_arm-linux-gnueabihf/bin/arm-linux-gnueabihf-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sun/myprojects/mygithub/rockchip_npu/telpo_AlgSDK_rk1109_intelif/src/telpo_algsdk.cpp -o CMakeFiles/telpoalgsdk.dir/src/telpo_algsdk.cpp.s

CMakeFiles/telpoalgsdk.dir/src/telpo_applications.cpp.o: CMakeFiles/telpoalgsdk.dir/flags.make
CMakeFiles/telpoalgsdk.dir/src/telpo_applications.cpp.o: ../src/telpo_applications.cpp
CMakeFiles/telpoalgsdk.dir/src/telpo_applications.cpp.o: CMakeFiles/telpoalgsdk.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sun/myprojects/mygithub/rockchip_npu/telpo_AlgSDK_rk1109_intelif/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/telpoalgsdk.dir/src/telpo_applications.cpp.o"
	/home/sun/Desktop/mydir/2-projects/arm/V61_sdk_out/prebuilts/gcc/linux-x86/arm/gcc-linaro-6.3.1-2017.05-x86_64_arm-linux-gnueabihf/bin/arm-linux-gnueabihf-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/telpoalgsdk.dir/src/telpo_applications.cpp.o -MF CMakeFiles/telpoalgsdk.dir/src/telpo_applications.cpp.o.d -o CMakeFiles/telpoalgsdk.dir/src/telpo_applications.cpp.o -c /home/sun/myprojects/mygithub/rockchip_npu/telpo_AlgSDK_rk1109_intelif/src/telpo_applications.cpp

CMakeFiles/telpoalgsdk.dir/src/telpo_applications.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/telpoalgsdk.dir/src/telpo_applications.cpp.i"
	/home/sun/Desktop/mydir/2-projects/arm/V61_sdk_out/prebuilts/gcc/linux-x86/arm/gcc-linaro-6.3.1-2017.05-x86_64_arm-linux-gnueabihf/bin/arm-linux-gnueabihf-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sun/myprojects/mygithub/rockchip_npu/telpo_AlgSDK_rk1109_intelif/src/telpo_applications.cpp > CMakeFiles/telpoalgsdk.dir/src/telpo_applications.cpp.i

CMakeFiles/telpoalgsdk.dir/src/telpo_applications.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/telpoalgsdk.dir/src/telpo_applications.cpp.s"
	/home/sun/Desktop/mydir/2-projects/arm/V61_sdk_out/prebuilts/gcc/linux-x86/arm/gcc-linaro-6.3.1-2017.05-x86_64_arm-linux-gnueabihf/bin/arm-linux-gnueabihf-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sun/myprojects/mygithub/rockchip_npu/telpo_AlgSDK_rk1109_intelif/src/telpo_applications.cpp -o CMakeFiles/telpoalgsdk.dir/src/telpo_applications.cpp.s

# Object files for target telpoalgsdk
telpoalgsdk_OBJECTS = \
"CMakeFiles/telpoalgsdk.dir/src/telpo_algsdk.cpp.o" \
"CMakeFiles/telpoalgsdk.dir/src/telpo_applications.cpp.o"

# External object files for target telpoalgsdk
telpoalgsdk_EXTERNAL_OBJECTS =

../lib/libtelpoalgsdk.so.1.0: CMakeFiles/telpoalgsdk.dir/src/telpo_algsdk.cpp.o
../lib/libtelpoalgsdk.so.1.0: CMakeFiles/telpoalgsdk.dir/src/telpo_applications.cpp.o
../lib/libtelpoalgsdk.so.1.0: CMakeFiles/telpoalgsdk.dir/build.make
../lib/libtelpoalgsdk.so.1.0: CMakeFiles/telpoalgsdk.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/sun/myprojects/mygithub/rockchip_npu/telpo_AlgSDK_rk1109_intelif/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX shared library ../lib/libtelpoalgsdk.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/telpoalgsdk.dir/link.txt --verbose=$(VERBOSE)
	$(CMAKE_COMMAND) -E cmake_symlink_library ../lib/libtelpoalgsdk.so.1.0 ../lib/libtelpoalgsdk.so.1.0 ../lib/libtelpoalgsdk.so

../lib/libtelpoalgsdk.so: ../lib/libtelpoalgsdk.so.1.0
	@$(CMAKE_COMMAND) -E touch_nocreate ../lib/libtelpoalgsdk.so

# Rule to build all files generated by this target.
CMakeFiles/telpoalgsdk.dir/build: ../lib/libtelpoalgsdk.so
.PHONY : CMakeFiles/telpoalgsdk.dir/build

CMakeFiles/telpoalgsdk.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/telpoalgsdk.dir/cmake_clean.cmake
.PHONY : CMakeFiles/telpoalgsdk.dir/clean

CMakeFiles/telpoalgsdk.dir/depend:
	cd /home/sun/myprojects/mygithub/rockchip_npu/telpo_AlgSDK_rk1109_intelif/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/sun/myprojects/mygithub/rockchip_npu/telpo_AlgSDK_rk1109_intelif /home/sun/myprojects/mygithub/rockchip_npu/telpo_AlgSDK_rk1109_intelif /home/sun/myprojects/mygithub/rockchip_npu/telpo_AlgSDK_rk1109_intelif/build /home/sun/myprojects/mygithub/rockchip_npu/telpo_AlgSDK_rk1109_intelif/build /home/sun/myprojects/mygithub/rockchip_npu/telpo_AlgSDK_rk1109_intelif/build/CMakeFiles/telpoalgsdk.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/telpoalgsdk.dir/depend


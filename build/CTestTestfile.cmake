# CMake generated Testfile for 
# Source directory: C:/Users/sirap/OneDrive/Documents/New project/tensoriit
# Build directory: C:/Users/sirap/OneDrive/Documents/New project/tensoriit/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[lexer]=] "C:/Users/sirap/OneDrive/Documents/New project/tensoriit/build/test_lexer.exe")
set_tests_properties([=[lexer]=] PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/sirap/OneDrive/Documents/New project/tensoriit/CMakeLists.txt;58;add_test;C:/Users/sirap/OneDrive/Documents/New project/tensoriit/CMakeLists.txt;0;")
add_test([=[parser]=] "C:/Users/sirap/OneDrive/Documents/New project/tensoriit/build/test_parser.exe")
set_tests_properties([=[parser]=] PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/sirap/OneDrive/Documents/New project/tensoriit/CMakeLists.txt;59;add_test;C:/Users/sirap/OneDrive/Documents/New project/tensoriit/CMakeLists.txt;0;")
add_test([=[codegen]=] "C:/Users/sirap/OneDrive/Documents/New project/tensoriit/build/test_codegen.exe")
set_tests_properties([=[codegen]=] PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/sirap/OneDrive/Documents/New project/tensoriit/CMakeLists.txt;60;add_test;C:/Users/sirap/OneDrive/Documents/New project/tensoriit/CMakeLists.txt;0;")

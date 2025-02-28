# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "XLink")

include(XLink.cmake)

add_library(${TARGET_NAME} STATIC ${XLINK_SOURCES})

if(NOT WIN32)
    target_include_directories(${TARGET_NAME}
            PRIVATE
            "${LIBUSB_INCLUDE_DIR}")

    target_link_libraries(${TARGET_NAME}
            PUBLIC
            Threads::Threads
            ${LIBUSB_LIBRARY})
endif()

target_include_directories(${TARGET_NAME} PRIVATE ${XLINK_PLATFORM_INCLUDE})
foreach(dir IN LISTS XLINK_INCLUDE)
    target_include_directories(${TARGET_NAME} PUBLIC $<BUILD_INTERFACE:${dir}>)
endforeach()

target_compile_definitions(${TARGET_NAME}
        PRIVATE
        __PC__
        HAVE_STRUCT_TIMESPEC
        _CRT_SECURE_NO_WARNINGS
        USE_USB_VSC
        )

if (ENABLE_MYRIAD_NO_BOOT)
    target_compile_definitions(${TARGET_NAME}
            PRIVATE
            NO_BOOT)
endif()

set_property(TARGET ${TARGET_NAME} PROPERTY C_STANDARD 99)

openvino_developer_export_targets(COMPONENT vpu TARGETS ${TARGET_NAME})
ov_install_static_lib(${TARGET_NAME} myriad)

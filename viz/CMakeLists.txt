set(__install_dest share/exago/viz)

set(EXAGO_VIZ_SUBDIRS backend components data geo_data images src)

set(EXAGO_VIZ_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/exago-logo.png
    ${CMAKE_CURRENT_SOURCE_DIR}/favicon.png
    ${CMAKE_CURRENT_SOURCE_DIR}/geninputfile.py
    ${CMAKE_CURRENT_SOURCE_DIR}/index.css
    ${CMAKE_CURRENT_SOURCE_DIR}/index.html
    ${CMAKE_CURRENT_SOURCE_DIR}/package.json
    ${CMAKE_CURRENT_SOURCE_DIR}/tsconfig.json
    ${CMAKE_CURRENT_SOURCE_DIR}/tsconfig.node.json
    ${CMAKE_CURRENT_SOURCE_DIR}/vite.config.ts
)

install(DIRECTORY ${EXAGO_VIZ_SUBDIRS} DESTINATION ${__install_dest})
install(FILES ${EXAGO_VIZ_FILES} DESTINATION ${__install_dest})

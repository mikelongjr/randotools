#!/bin/bash
# build_rpm.sh - Build the RealESRGAN Upscaler RPM package

set -e

# Configuration
PACKAGE_NAME="realesrgan-upscaler"
VERSION="2.0.0"
SPEC_FILE="realesrgan-upscaler.spec"
BUILD_DIR="rpmbuild"
STAGING_DIR="$BUILD_DIR/SOURCES/$PACKAGE_NAME-$VERSION"

# Create RPM build directory structure
rm -rf "$STAGING_DIR"
mkdir -p "$BUILD_DIR"/{SOURCES,SPECS,BUILD,RPMS,SRPMS} "$STAGING_DIR"

# Create a source tarball for the RPM build process
# We include the application code, the wrapper, the desktop file, and the icon
echo "Creating source tarball..."
cp -r upscaler requirements.txt realesrgan-wrapper.sh realesrgan-upscaler.desktop realesrgan-upscaler.svg "$STAGING_DIR"/
tar -czf "$BUILD_DIR/SOURCES/$PACKAGE_NAME-$VERSION.tar.gz" \
    -C "$BUILD_DIR/SOURCES" \
    "$PACKAGE_NAME-$VERSION"

# Copy the spec file to the SPECS directory
cp "$SPEC_FILE" "$BUILD_DIR/SPECS/"

# Build the RPM
echo "Building RPM package..."
rpmbuild -ba "$BUILD_DIR/SPECS/$SPEC_FILE" --define "_topdir $(pwd)/$BUILD_DIR"

echo "--------------------------------------------------------------------------------"
echo "RPM build complete!"
echo "Package located at: $BUILD_DIR/RPMS/x86_64/"
echo "--------------------------------------------------------------------------------"

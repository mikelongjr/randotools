Name:           realesrgan-upscaler
Version:        2.0.0
Release:        1%{?dist}
Summary:        PyQt6-based RealESRGAN image upscaler with GPU support

License:        MIT
URL:            https://github.com/mikelongjr/randotools
Vendor:         mikelongjr
BuildArch:      noarch

Source0:        %{name}-%{version}.tar.gz

BuildRequires:  rpm-build
Requires:       python3, ffmpeg, python3-qt6, libGL, mesa-libGL

%description
RealESRGAN Upscaler is a GUI application for upscaling images using AI.
It supports NVIDIA and AMD GPUs via PyTorch. This RPM installs the app files
and launcher; use fedora_setup.sh or a project virtualenv for ML dependencies.

%prep
%setup -q

%build
# No compilation needed for Python.

%install
# Copy the application source and requirements
mkdir -p %{buildroot}/opt/realesrgan-upscaler
cp -r upscaler %{buildroot}/opt/realesrgan-upscaler/
cp requirements.txt %{buildroot}/opt/realesrgan-upscaler/

# Copy the wrapper script
cp realesrgan-wrapper.sh %{buildroot}/usr/bin/realesrgan-upscaler
chmod +x %{buildroot}/usr/bin/realesrgan-upscaler

# Copy the desktop file
cp realesrgan-upscaler.desktop %{buildroot}/usr/share/applications/

# Copy the icon
cp realesrgan-upscaler.svg %{buildroot}/usr/share/icons/hicolor/scalable/apps/

%postun
if [ $1 -eq 0 ]; then
    # Only remove if it's a full uninstall
    rm -rf /opt/realesrgan-upscaler
fi

%files
/opt/realesrgan-upscaler
/usr/bin/realesrgan-upscaler
/usr/share/applications/realesrgan-upscaler.desktop
/usr/share/icons/hicolor/scalable/apps/realesrgan-upscaler.svg

%changelog
* Wed Apr 25 2026 mikelongjr <mikelongjr@example.com> - 2.0.0-1
- Initial RPM package release

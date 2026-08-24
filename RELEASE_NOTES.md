### 0.3.0

[MINOR] `Analyzer.init()` accepts `usb_serial` to open a specific QA402/QA403 by its USB serial number, so a PC with several analyzers can drive a chosen one. The argument is last and optional; without it the first analyzer found is opened, exactly as before.

### 0.2.2

[PATCH] Allow reliable streaming for higher sample rates (`96kHz`, `192kHz`).

### 0.2.1

[PATCH] Restore removed `Analyzer.send_receive` functionality.

### 0.2.0

[MINOR] Add endless audio capturing 

### 0.1.2

[PATCH] Fix usb teardown deadlock.

### 0.1.1

[PATCH] Fix missing dependency.

### 0.1.0

[MINOR] Initial release as PyQa40x-i2c fork. Unlimited audio samples capture added.

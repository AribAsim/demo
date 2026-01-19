# Roadmap: System-Wide Content Filtering for SafeBrowse

This document outlines the strategic roadmap for evolving SafeBrowse from a standalone browser application into a system-wide parental control agent using Android Accessibility Services.

## 🎯 Objective
To monitor and filter content across **ALL** applications (Chrome, Instagram, YouTube, TikTok, etc.) on the child's device, enforcing parental controls at the operating system level without requiring root access.

---

## 🏗️ Architecture Override

### Current Architecture (Browser-Based)
*   **Scope:** Monitoring only happens inside the SafeBrowse app's `WebView`.
*   **Limitation:** Child can simply open Chrome or Play Store to bypass filters.

### Target Architecture (System-Wide)
*   **Scope:** An `AccessibilityService` runs in the background, intercepting UI events from every active app.
*   **Mechanism:**
    1.  **Read:** Extract text from screen nodes (URL bars, chat messages, post descriptions).
    2.  **Analyze:**
        *   *Fast Path:* Local checking against a cache of banned keywords/domains.
        *   *Deep Path:* Send snippets to Local Python Backend (if on Wi-Fi) or Cloud Backend (Production) for AI analysis.
    3.  **Act:** If harmful content is found, instantly trigger the `GLOBAL_ACTION_BACK` or draw a full-screen "Blocked" overlay window over the offending app.

---

## 📅 Implementation Phases

### Phase 1: Engineering Setup (Transition to Native)
*The goal is to prepare the React Native environment for custom native code injection.*

1.  **Eject from Expo Go**:
    *   Transition project to **Expo Dev Client**.
    *   Configure `eas.json` for building local Android binaries (`.apk`).
2.  **Android Setup**:
    *   Set up Android Studio SDKs and JDK 17.
    *   Verify ability to compile the current app into a standalone APK (`npx expo run:android`).

### Phase 2: The Accessibility Engine (Native Module)
*The core engine that runs silently in the background.*

1.  **Config Plugin Development**:
    *   Create `withAccessibilityService.js`: An Expo Config Plugin to modify `AndroidManifest.xml` automatically.
    *   Define `service_config.xml`: Specify that we want to listen to `typeViewTextSelectionChanged`, `typeWindowContentChanged`.
2.  **Java/Kotlin Service Implementation**:
    *   Create `SafeBrowseAccessibilityService.java`.
    *   Implement logic to traverse `AccessibilityNodeInfo` tree.
    *   Extract text from `EditText` (Input), `TextView` (Display), and specifically `URL Bar` IDs for Chrome/Samsung Internet.
3.  **Bridge to React Native**:
    *   Implement JSI or Native Module bridge to send data from Java Service -> React Native JS.
    *   *Why?* To use our existing JS/Typescript logic for profile management and logging.

### Phase 3: Blocking Mechanics
*How we enforce the rules.*

1.  **Overlay Window**:
    *   Create a native Android View (XML Layout) for the "Blocked" screen.
    *   Request `SYSTEM_ALERT_WINDOW` permission ("Draw over other apps").
2.  **Action Trigger**:
    *   If Keywords Found -> Inflate Overlay View immediately.
    *   If Clean -> Remove Overlay (if present).
3.  **App Locking (Bonus)**:
    *   Detect package names (e.g., `com.instagram.android`).
    *   If user < 16, instantly block the package launch.

### Phase 4: Integration with Existing Brain
*Connecting the new eyes to the old brain.*

1.  **Profile Synchronization**:
    *   The Background Service needs to know *which* profile is active.
    *   Store active profile rules (keywords, age limits) in Android `SharedPreferences` (accessible by both React Native UI and Java Service).
2.  **Offline/Online Logic**:
    *   Update `server.py` to accept simple text blobs from the service.
    *   *Constraint Check:* Accessibility service cannot upload images efficiently. Focus will be 90% text/URL based for system-wide monitoring.

---

## 🛠️ Workflow for the Team

### 1. Backend Team (Python)
*   **Action:** No major rewrite.
*   **Task:** Create a lightweight endpoints for "Batch Text Classification" to handle rapid-fire text coming from the screen reader.

### 2. Mobile Team (React Native)
*   **Action:** Heavy lifting.
*   **Task 1:** Set up the 'Dev Client' workflow (forget Expo Go app).
*   **Task 2:** Write the Java/Kotlin sensor code.
*   **Task 3:** Create the "Permission onboarding" flow (guiding parents to enable Accessibility permissions in Settings).

### 3. QA / Testing
*   **Challenge:** You cannot automate testing for this comfortably.
*   **Task:** Manual testing on physical Android devices is mandatory. Emulators often fail to replicate exact Accessibility behavior.

---

## ⚠️ Key Risks & Mitigations

| Risk | Mitigation |
| :--- | :--- |
| **Battery Drain** | Limit scanning frequency (debounce events by 500ms). Only scan specific apps (allowlist/blocklist). |
| **Privacy Concerns** | Do NOT log keystrokes (passwords). Only scan View text. Clearly state strictly local analysis in privacy policy. |
| **Google Play Rejection** | Must forcefully justify "Parental Control" use case. Use the `isAccessibilityTool` flag in manifest. |
| **App Killing by OS** | Implement a "Foreground Service" notification ("SafeBrowse is protecting you") to prevent Android from killing the process. |

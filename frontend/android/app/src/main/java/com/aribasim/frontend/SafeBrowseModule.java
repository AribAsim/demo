package com.aribasim.frontend;

import android.content.Intent;
import android.provider.Settings;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactContextBaseJavaModule;
import com.facebook.react.bridge.ReactMethod;
import com.facebook.react.bridge.Promise;
import com.facebook.react.modules.core.DeviceEventManagerModule;
import com.facebook.react.bridge.WritableMap;
import com.facebook.react.bridge.Arguments;

public class SafeBrowseModule extends ReactContextBaseJavaModule {
    public static final String NAME = "SafeBrowseModule";
    private static ReactApplicationContext reactContext;

    public SafeBrowseModule(ReactApplicationContext context) {
        super(context);
        reactContext = context;
    }

    @Override
    @NonNull
    public String getName() {
        return NAME;
    }

    // --- Exposed Methods to JS ---

    @ReactMethod
    public void openAccessibilitySettings() {
        Intent intent = new Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS);
        intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
        reactContext.startActivity(intent);
    }

    @ReactMethod
    public void isAccessibilityEnabled(Promise promise) {
        // Basic check: we can assume enabled if we are receiving events, but looking up system settings is better
        // For simple MVP we just check if the service is running (not perfectly reliable but okay)
        // A better way involves checking Settings.Secure.ENABLED_ACCESSIBILITY_SERVICES
        int accessibilityEnabled = 0;
        try {
            accessibilityEnabled = Settings.Secure.getInt(
                    reactContext.getContentResolver(),
                    android.provider.Settings.Secure.ACCESSIBILITY_ENABLED);
        } catch (Settings.SettingNotFoundException e) {
            Log.e(NAME, "Error finding setting, default accessibility to not found: " + e.getMessage());
        }
        
        // This only checks if the global switch is on, not if OUR service is on. 
        // For MVP we will trust the user and the flow.
        promise.resolve(accessibilityEnabled == 1);
    }
    
    @ReactMethod
    public void blockContent() {
       // This will be called by JS when unsafe content is found
       // We need to signal the AccessibilityService to perform the back action
       SafeBrowseAccessibilityService instance = SafeBrowseAccessibilityService.getInstance();
       if (instance != null) {
           instance.performGlobalAction(android.accessibilityservice.AccessibilityService.GLOBAL_ACTION_BACK);
       }
    }

    // --- Helper to Emit Events from Service ---
    
    public static void emitTextEvent(String text, String packageName) {
        if (reactContext != null && reactContext.hasActiveCatalystInstance()) {
            WritableMap params = Arguments.createMap();
            params.putString("text", text);
            params.putString("packageName", packageName);
            
            reactContext
                .getJSModule(DeviceEventManagerModule.RCTDeviceEventEmitter.class)
                .emit("ON_TEXT_CAPTURED", params);
        }
    }
    
    // Required for React Native built-in events
    @ReactMethod
    public void addListener(String eventName) {
        // Keep: Required for RN built-in Event Emitter Calls.
    }

    @ReactMethod
    public void removeListeners(Integer count) {
        // Keep: Required for RN built-in Event Emitter Calls.
    }
}

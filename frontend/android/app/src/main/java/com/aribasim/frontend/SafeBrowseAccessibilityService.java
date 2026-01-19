package com.aribasim.frontend;

import android.accessibilityservice.AccessibilityService;
import android.view.accessibility.AccessibilityEvent;
import android.view.accessibility.AccessibilityNodeInfo;
import android.util.Log;
import java.util.List;

public class SafeBrowseAccessibilityService extends AccessibilityService {
    private static final String TAG = "SafeBrowseService";
    private static SafeBrowseAccessibilityService instance;

    public static SafeBrowseAccessibilityService getInstance() {
        return instance;
    }

    @Override
    protected void onServiceConnected() {
        super.onServiceConnected();
        instance = this;
        Log.d(TAG, "SafeBrowse Service Connected");
    }

    @Override
    public void onAccessibilityEvent(AccessibilityEvent event) {
        if (event == null) return;
        
        // Skip our own app to avoid infinite loops if we display logs
        if (event.getPackageName() != null && event.getPackageName().toString().contains("com.aribasim.frontend")) {
             return;
        }

        // Just capturing generic text for now on specific event types
        if (event.getEventType() == AccessibilityEvent.TYPE_VIEW_TEXT_SELECTION_CHANGED ||
            event.getEventType() == AccessibilityEvent.TYPE_WINDOW_CONTENT_CHANGED || 
            event.getEventType() == AccessibilityEvent.TYPE_VIEW_FOCUSED) {
            
            // Text Extraction Logic
            if (event.getText() != null && !event.getText().isEmpty()) {
                String capturedText = event.getText().toString();
                String pkg = event.getPackageName() != null ? event.getPackageName().toString() : "unknown";
                
                // Emit to React Native
                // We debounce slightly or clean the text in a real app
                SafeBrowseModule.emitTextEvent(capturedText, pkg);
            }
        }
    }

    @Override
    public void onInterrupt() {
        Log.e(TAG, "SafeBrowse Service Interrupted");
        instance = null;
    }
}

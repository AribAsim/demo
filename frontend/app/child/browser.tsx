import React, { useState, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  TextInput,
  Alert,
  Modal,
  ActivityIndicator,
  KeyboardAvoidingView,
  Platform,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { WebView } from 'react-native-webview';
import { useRouter } from 'expo-router';
import { useAuth } from '../../contexts/AuthContext';
import { useAppMode } from '../../contexts/AppModeContext';
import axios from 'axios';
import Constants from 'expo-constants';

const API_URL = Constants.expoConfig?.extra?.EXPO_PUBLIC_BACKEND_URL || process.env.EXPO_PUBLIC_BACKEND_URL;

export default function ChildBrowser() {
  const { user, token } = useAuth();
  const { setMode, selectedProfile } = useAppMode();
  const router = useRouter();
  const webViewRef = useRef<WebView>(null);

  const [url, setUrl] = useState('https://www.google.com');
  const [currentUrl, setCurrentUrl] = useState('https://www.google.com');
  const [canGoBack, setCanGoBack] = useState(false);
  const [canGoForward, setCanGoForward] = useState(false);
  const [loading, setLoading] = useState(false);
  const [blocked, setBlocked] = useState(false);
  const [blockReason, setBlockReason] = useState('');
  const [exitModalVisible, setExitModalVisible] = useState(false);
  const [pin, setPin] = useState('');

  const analyzeUrl = async (targetUrl: string) => {
    try {
      const response = await axios.post(
        `${API_URL}/api/content/analyze`,
        {
          profile_id: selectedProfile,
          content_type: 'url',
          content: targetUrl,
          context: targetUrl,
        },
        {
          headers: { Authorization: `Bearer ${token}` },
        }
      );

      if (response.data.blocked) {
        setBlocked(true);
        setBlockReason(response.data.reasons.join(', '));
        return false;
      }

      return true;
    } catch (error) {
      console.error('Error analyzing URL:', error);
      return true; // Allow on error to not break browsing
    }
  };

  const handleNavigate = async () => {
    let targetUrl = url.trim();

    if (!targetUrl) {
      return;
    }

    // Add https:// if no protocol specified
    if (!targetUrl.startsWith('http://') && !targetUrl.startsWith('https://')) {
      targetUrl = 'https://' + targetUrl;
    }

    setLoading(true);
    const isSafe = await analyzeUrl(targetUrl);
    setLoading(false);

    if (isSafe) {
      setCurrentUrl(targetUrl);
      setBlocked(false);
      webViewRef.current?.reload();
    }
  };

  const handleWebViewNavigationStateChange = async (navState: any) => {
    setCanGoBack(navState.canGoBack);
    setCanGoForward(navState.canGoForward);

    if (navState.url !== currentUrl) {
      const isSafe = await analyzeUrl(navState.url);
      if (!isSafe) {
        webViewRef.current?.stopLoading();
        return false;
      }
    }
  };

  const handleWebViewMessage = async (event: any) => {
    try {
      const data = JSON.parse(event.nativeEvent.data);

      if (data.type === 'text' || data.type === 'image') {
        const response = await axios.post(
          `${API_URL}/api/content/analyze`,
          {
            profile_id: selectedProfile,
            content_type: data.type,
            content: data.content,
            context: currentUrl,
          },
          {
            headers: { Authorization: `Bearer ${token}` },
          }
        );

        if (response.data.blocked) {
          setBlocked(true);
          setBlockReason(response.data.reasons.join(', '));
        }
      }
    } catch (error) {
      console.error('Error handling webview message:', error);
    }
  };

  const handleExitChildMode = () => {
    if (!user?.pin) {
      Alert.alert('Error', 'PIN not set. Please contact parent.');
      return;
    }
    setExitModalVisible(true);
  };

  const verifyPinAndExit = () => {
    if (pin === user?.pin) {
      setMode('parent');
      setExitModalVisible(false);
      setPin('');
      router.replace('/parent/dashboard');
    } else {
      Alert.alert('Error', 'Incorrect PIN');
      setPin('');
    }
  };

  const handleUnblock = () => {
    setBlocked(false);
    setBlockReason('');
  };

  // Inject script to intercept content
  const injectedJavaScript = `
    (function() {
      const processedImages = new Set();
      const processedTexts = new Set();

      function scanContent() {
        // 1. Scan Text (Titles, Headers, Body)
        const bodyText = document.body.innerText.substring(0, 1000);
        if (bodyText && !processedTexts.has(bodyText)) {
          window.ReactNativeWebView.postMessage(JSON.stringify({
            type: 'text',
            content: bodyText
          }));
          processedTexts.add(bodyText);
        }

        // 2. Scan Images (Regular tags)
        const images = document.getElementsByTagName('img');
        for (let i = 0; i < images.length; i++) {
          const src = images[i].src;
          if (src && !processedImages.has(src)) {
            // Process if it looks like an image or data URI
            if (src.startsWith('data:') || src.includes('image') || src.match(/\\.(jpg|jpeg|png|webp|gif|avif)/i)) {
              window.ReactNativeWebView.postMessage(JSON.stringify({
                type: 'image',
                content: src
              }));
              processedImages.add(src);
            }
          }
        }

        // 3. Scan for CSS Background Images (Common in modern sites)
        const elementsWithBg = document.querySelectorAll('[style*="background-image"]');
        elementsWithBg.forEach(el => {
          const style = window.getComputedStyle(el);
          const bg = style.backgroundImage;
          if (bg && bg !== 'none' && bg.startsWith('url(')) {
            const url = bg.slice(4, -1).replace(/["']/g, "");
            if (url && !processedImages.has(url)) {
              window.ReactNativeWebView.postMessage(JSON.stringify({
                type: 'image',
                content: url
              }));
              processedImages.add(url);
            }
          }
        });

        // 4. Scan for Videos (Metadata)
        const videos = document.getElementsByTagName('video');
        if (videos.length > 0 && !processedTexts.has('video_detected')) {
          const videoTags = Array.from(document.querySelectorAll('meta[name*="title"], meta[property*="title"]'))
            .map(m => m.content).join(' ');
          if (videoTags) {
            window.ReactNativeWebView.postMessage(JSON.stringify({
              type: 'text',
              content: "Video content title: " + videoTags
            }));
          }
          processedTexts.add('video_detected');
        }
      }

      // Monitor for changes
      const observer = new MutationObserver(() => {
        scanContent();
      });
      
      observer.observe(document.body, {
        childList: true,
        subtree: true,
        attributes: true,
        attributeFilter: ['src']
      });

      // Initial check
      setTimeout(scanContent, 1000);
      // Periodical check for dynamic content not caught by observer
      setInterval(scanContent, 3000);
    })();
    true;
  `;

  return (
    <SafeAreaView style={styles.container} edges={['top']}>
      <View style={styles.header}>
        <TouchableOpacity
          style={styles.exitButton}
          onPress={handleExitChildMode}
        >
          <Ionicons name="arrow-back" size={24} color="#ffffff" />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Safe Browser</Text>
        <View style={styles.headerRight}>
          <Ionicons name="shield-checkmark" size={24} color="#10b981" />
        </View>
      </View>

      <View style={styles.urlBar}>
        <TouchableOpacity
          onPress={() => webViewRef.current?.goBack()}
          disabled={!canGoBack}
          style={[styles.navButton, !canGoBack && styles.navButtonDisabled]}
        >
          <Ionicons name="chevron-back" size={20} color={canGoBack ? '#f1f5f9' : '#64748b'} />
        </TouchableOpacity>

        <TouchableOpacity
          onPress={() => webViewRef.current?.goForward()}
          disabled={!canGoForward}
          style={[styles.navButton, !canGoForward && styles.navButtonDisabled]}
        >
          <Ionicons name="chevron-forward" size={20} color={canGoForward ? '#f1f5f9' : '#64748b'} />
        </TouchableOpacity>

        <View style={styles.urlInputContainer}>
          <Ionicons name="globe" size={16} color="#64748b" style={styles.urlIcon} />
          <TextInput
            style={styles.urlInput}
            value={url}
            onChangeText={setUrl}
            onSubmitEditing={handleNavigate}
            placeholder="Enter URL or search..."
            placeholderTextColor="#64748b"
            autoCapitalize="none"
            autoCorrect={false}
          />
        </View>

        <TouchableOpacity onPress={handleNavigate} style={styles.goButton}>
          <Ionicons name="arrow-forward" size={20} color="#ffffff" />
        </TouchableOpacity>
      </View>

      {loading && (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#6366f1" />
          <Text style={styles.loadingText}>Checking content safety...</Text>
        </View>
      )}

      {blocked ? (
        <View style={styles.blockedContainer}>
          <View style={styles.blockedContent}>
            <Ionicons name="shield-outline" size={80} color="#ef4444" />
            <Text style={styles.blockedTitle}>Content Blocked</Text>
            <Text style={styles.blockedReason}>{blockReason}</Text>
            <Text style={styles.blockedMessage}>
              This content has been blocked to keep you safe. If you think this is a mistake,
              ask your parent to review it.
            </Text>
            <TouchableOpacity style={styles.unblockButton} onPress={handleUnblock}>
              <Text style={styles.unblockButtonText}>Go Back</Text>
            </TouchableOpacity>
          </View>
        </View>
      ) : (
        <WebView
          ref={webViewRef}
          source={{ uri: currentUrl }}
          style={styles.webview}
          onNavigationStateChange={handleWebViewNavigationStateChange}
          onMessage={handleWebViewMessage}
          injectedJavaScript={injectedJavaScript}
          javaScriptEnabled
          domStorageEnabled
          startInLoadingState
          renderLoading={() => (
            <View style={styles.loadingContainer}>
              <ActivityIndicator size="large" color="#6366f1" />
            </View>
          )}
        />
      )}

      <Modal visible={exitModalVisible} animationType="slide" transparent>
        <KeyboardAvoidingView
          behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
          style={styles.modalContainer}
        >
          <View style={styles.modalContent}>
            <View style={styles.modalHeader}>
              <Text style={styles.modalTitle}>Enter Parent PIN</Text>
              <TouchableOpacity onPress={() => setExitModalVisible(false)}>
                <Ionicons name="close" size={24} color="#94a3b8" />
              </TouchableOpacity>
            </View>

            <Text style={styles.modalDescription}>
              Enter your parent&apos;s PIN to exit Safe Browser
            </Text>

            <TextInput
              style={styles.pinInput}
              value={pin}
              onChangeText={setPin}
              placeholder="Enter 4-digit PIN"
              placeholderTextColor="#64748b"
              keyboardType="number-pad"
              maxLength={4}
              secureTextEntry
            />

            <TouchableOpacity style={styles.verifyButton} onPress={verifyPinAndExit}>
              <Text style={styles.verifyButtonText}>Verify & Exit</Text>
            </TouchableOpacity>
          </View>
        </KeyboardAvoidingView>
      </Modal>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0f172a',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingVertical: 12,
    backgroundColor: '#1e293b',
    borderBottomWidth: 1,
    borderBottomColor: '#334155',
  },
  exitButton: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: '#334155',
    alignItems: 'center',
    justifyContent: 'center',
  },
  headerTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#f1f5f9',
  },
  headerRight: {
    width: 40,
    height: 40,
    alignItems: 'center',
    justifyContent: 'center',
  },
  urlBar: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 12,
    backgroundColor: '#1e293b',
    borderBottomWidth: 1,
    borderBottomColor: '#334155',
  },
  navButton: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: '#334155',
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 8,
  },
  navButtonDisabled: {
    opacity: 0.4,
  },
  urlInputContainer: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#0f172a',
    borderRadius: 20,
    paddingHorizontal: 12,
    height: 36,
    borderWidth: 1,
    borderColor: '#334155',
  },
  urlIcon: {
    marginRight: 8,
  },
  urlInput: {
    flex: 1,
    color: '#f1f5f9',
    fontSize: 14,
  },
  goButton: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: '#6366f1',
    alignItems: 'center',
    justifyContent: 'center',
    marginLeft: 8,
  },
  webview: {
    flex: 1,
    backgroundColor: '#ffffff',
  },
  loadingContainer: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#0f172a',
  },
  loadingText: {
    color: '#94a3b8',
    marginTop: 16,
    fontSize: 16,
  },
  blockedContainer: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#0f172a',
    padding: 24,
  },
  blockedContent: {
    alignItems: 'center',
    maxWidth: 400,
  },
  blockedTitle: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#ef4444',
    marginTop: 24,
  },
  blockedReason: {
    fontSize: 16,
    color: '#f59e0b',
    marginTop: 12,
    textAlign: 'center',
  },
  blockedMessage: {
    fontSize: 16,
    color: '#94a3b8',
    marginTop: 16,
    textAlign: 'center',
    lineHeight: 24,
  },
  unblockButton: {
    backgroundColor: '#6366f1',
    paddingHorizontal: 32,
    paddingVertical: 16,
    borderRadius: 12,
    marginTop: 32,
  },
  unblockButtonText: {
    color: '#ffffff',
    fontSize: 16,
    fontWeight: '600',
  },
  modalContainer: {
    flex: 1,
    justifyContent: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    padding: 24,
  },
  modalContent: {
    backgroundColor: '#1e293b',
    borderRadius: 24,
    padding: 24,
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  modalTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#f1f5f9',
  },
  modalDescription: {
    fontSize: 14,
    color: '#94a3b8',
    marginBottom: 24,
    lineHeight: 20,
  },
  pinInput: {
    backgroundColor: '#0f172a',
    borderRadius: 12,
    padding: 16,
    color: '#f1f5f9',
    fontSize: 24,
    textAlign: 'center',
    borderWidth: 1,
    borderColor: '#334155',
    letterSpacing: 8,
  },
  verifyButton: {
    backgroundColor: '#6366f1',
    borderRadius: 12,
    height: 56,
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 16,
  },
  verifyButtonText: {
    color: '#ffffff',
    fontSize: 18,
    fontWeight: '600',
  },
});

import React, { useState, useRef, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  TextInput,
  ScrollView,
  Animated,
  Dimensions,
  Platform,
  StatusBar,
  Modal,
  KeyboardAvoidingView,
  Alert,
  BackHandler,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { WebView } from 'react-native-webview';
import { useRouter } from 'expo-router';
import { useAuth } from '../../contexts/AuthContext';
import { useAppMode } from '../../contexts/AppModeContext';
import axios from 'axios';
import Constants from 'expo-constants';

const { width, height } = Dimensions.get('window');
const API_URL = Constants.expoConfig?.extra?.EXPO_PUBLIC_BACKEND_URL || process.env.EXPO_PUBLIC_BACKEND_URL;

interface Shortcut {
  icon: keyof typeof Ionicons.glyphMap;
  label: string;
  color: string;
  url: string;
}

interface Tab {
  id: number;
  url: string;
  title: string;
  favicon: string;
}

export default function SafeBrowseBrowser() {
  const { user, token } = useAuth();
  const { setMode, selectedProfile } = useAppMode();
  const router = useRouter();
  const webViewRef = useRef<WebView>(null);
  const scrollRef = useRef<ScrollView>(null);

  // Core States
  const [url, setUrl] = useState('https://www.google.com');
  const [currentUrl, setCurrentUrl] = useState('https://www.google.com');
  const [canGoBack, setCanGoBack] = useState(false);
  const [canGoForward, setCanGoForward] = useState(false);
  const [childName, setChildName] = useState('Child');

  // UI States
  const [loading, setLoading] = useState(false);
  const [blocked, setBlocked] = useState(false);
  const [blockReason, setBlockReason] = useState('');
  const [loadProgress, setLoadProgress] = useState(0);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [isScanning, setIsScanning] = useState(false);
  const [maturityLevel, setMaturityLevel] = useState('strict');
  const [blockedCount, setBlockedCount] = useState(3);
  const [screenTime, setScreenTime] = useState(45);
  const [safetyScore, setSafetyScore] = useState(95);

  // Modals
  const [exitModalVisible, setExitModalVisible] = useState(false);
  const [tabsModalVisible, setTabsModalVisible] = useState(false);
  const [pin, setPin] = useState('');

  // Tabs
  const [tabs, setTabs] = useState<Tab[]>([
    { id: 1, url: 'https://www.google.com', title: 'Google', favicon: '🌐' }
  ]);
  const [activeTab, setActiveTab] = useState(1);

  // Animation
  const pulseAnim = useRef(new Animated.Value(1)).current;
  const shieldScale = useRef(new Animated.Value(0)).current;

  // Shield pulse animation
  useEffect(() => {
    if (isScanning) {
      Animated.loop(
        Animated.sequence([
          Animated.timing(pulseAnim, {
            toValue: 1.2,
            duration: 600,
            useNativeDriver: true,
          }),
          Animated.timing(pulseAnim, {
            toValue: 1,
            duration: 600,
            useNativeDriver: true,
          }),
        ])
      ).start();
    } else {
      Animated.spring(pulseAnim, {
        toValue: 1,
        useNativeDriver: true,
      }).start();
    }
  }, [isScanning]);

  // Show shield animation on mount
  useEffect(() => {
    Animated.spring(shieldScale, {
      toValue: 1,
      tension: 50,
      friction: 7,
      useNativeDriver: true,
    }).start();
  }, []);

  // Fetch Child Name
  useEffect(() => {
    const fetchProfile = async () => {
      try {
        if (selectedProfile && token) {
          const response = await axios.get(`${API_URL}/api/profiles/${selectedProfile}`, {
            headers: { Authorization: `Bearer ${token}` }
          });
          if (response.data && response.data.name) {
            setChildName(response.data.name);
            setMaturityLevel(response.data.maturity_level || 'strict');
          }
        }
      } catch (error) {
        console.error('Error fetching profile:', error);
      }
    };
    fetchProfile();
  }, [selectedProfile, token]);

  // Handle Hardware Back Button
  useEffect(() => {
    const backAction = () => {
      // Navigate back in WebView if possible
      if (canGoBack && webViewRef.current) {
        webViewRef.current.goBack();
        return true;
      }

      // Otherwise ask for PIN to exit
      handleExitChildMode();
      return true;
    };

    const backHandler = BackHandler.addEventListener(
      'hardwareBackPress',
      backAction
    );

    return () => backHandler.remove();
  }, [canGoBack]); // Re-bind when canGoBack changes so we have latest state

  const shortcuts: Shortcut[] = [
    { icon: 'home', label: 'Home', color: '#6366f1', url: 'https://www.pbskids.org' },
    { icon: 'book', label: 'Learn', color: '#10b981', url: 'https://www.khanacademy.org' },
    { icon: 'game-controller', label: 'Games', color: '#a855f7', url: 'https://www.coolmathgames.com' },
    { icon: 'color-palette', label: 'Create', color: '#ec4899', url: 'https://scratch.mit.edu' },
    { icon: 'flask', label: 'Science', color: '#f59e0b', url: 'https://www.natgeokids.com' },
    { icon: 'musical-notes', label: 'Music', color: '#06b6d4', url: 'https://musiclab.chromeexperiments.com' },
    { icon: 'videocam', label: 'Videos', color: '#ef4444', url: 'https://www.youtube.com/kids' },
    { icon: 'globe', label: 'Explore', color: '#8b5cf6', url: 'https://www.worldometers.info' },
    { icon: 'calculator', label: 'Math', color: '#f97316', url: 'https://www.mathplayground.com' },
  ];

  const suggestions = [
    'dinosaurs facts for kids',
    'fun math games',
    'science experiments at home',
  ];

  // Content Analysis
  const analyzeUrl = async (targetUrl: string) => {
    try {
      if (!selectedProfile) return true; // Can't analyze without profile

      setIsScanning(true);
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

      setIsScanning(false);

      if (response.data.blocked) {
        setBlocked(true);
        setBlockReason(response.data.reasons.join(', '));
        setBlockedCount(prev => prev + 1);
        return false;
      }

      return true;
    } catch (error) {
      console.error('Error analyzing URL:', error);
      setIsScanning(false);
      return true; // Fail OPEN to keep browser usable
    }
  };

  const handleNavigate = async () => {
    let targetUrl = url.trim();

    if (!targetUrl) return;

    // Check if it's a search or a URL
    const isUrl = (targetUrl.startsWith('http://') || targetUrl.startsWith('https://')) || (targetUrl.includes('.') && !targetUrl.includes(' '));

    if (!isUrl) {
      targetUrl = `https://www.google.com/search?q=${encodeURIComponent(targetUrl)}`;
    } else if (!targetUrl.startsWith('http://') && !targetUrl.startsWith('https://')) {
      targetUrl = 'https://' + targetUrl;
    }

    setLoading(true);
    const isSafe = await analyzeUrl(targetUrl);
    setLoading(false);

    if (isSafe) {
      setCurrentUrl(targetUrl);
      setBlocked(false);
      setShowSuggestions(false);
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
        if (!selectedProfile) return;

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
          // Image violation logic
          if (data.type === 'image' && (maturityLevel === 'moderate' || maturityLevel === 'lenient')) {
            // Blurred blocking for moderate/lenient
            const script = `if(window.safeBrowseBlockImage) window.safeBrowseBlockImage('${data.id}'); true;`;
            webViewRef.current?.injectJavaScript(script);
            setBlockedCount(prev => prev + 1);
          } else {
            // Full page block for strict mode or text violations
            webViewRef.current?.stopLoading();
            setBlocked(true);
            setBlockReason(response.data.reasons.join(', '));
            setBlockedCount(prev => prev + 1);
          }
        }
      }
    } catch (error) {
      // Silent fail for background scans to not spam logs
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

  const handleShortcutPress = (shortcutUrl: string) => {
    setUrl(shortcutUrl);
    setCurrentUrl(shortcutUrl);
    setBlocked(false);
    webViewRef.current?.reload();
  };

  const addNewTab = () => {
    const newTab: Tab = {
      id: Date.now(),
      url: 'https://www.google.com',
      title: 'New Tab',
      favicon: '🌐',
    };
    setTabs([...tabs, newTab]);
    setActiveTab(newTab.id);
    setCurrentUrl(newTab.url);
  };

  const closeTab = (tabId: number) => {
    if (tabs.length === 1) return;

    const filtered = tabs.filter(tab => tab.id !== tabId);
    setTabs(filtered);

    if (activeTab === tabId && filtered.length > 0) {
      setActiveTab(filtered[0].id);
      setCurrentUrl(filtered[0].url);
    }
  };

  const injectedJavaScript = `
    (function() {
      const processedImages = new Set();
      const processedTexts = new Set();
      
      // Image blocking receiver
      if (!window.safeBrowseBlockImage) {
        window.safeBrowseBlockImage = function(id) {
          const img = document.querySelector('img[data-safebrowse-id="' + id + '"]');
          if (img) {
             img.style.filter = 'blur(60px) grayscale(100%)'; // Increased blur
             img.style.opacity = '0.3';
             img.style.border = '4px solid #ef4444'; // Red border
             img.style.pointerEvents = 'none';
          }
        };
      }
      
      // Debounce function to prevent spamming
      let timeout = null;

      function scanContent() {
        // Only scan significant text chunks (> 50 chars) to avoid random noise
        const bodyText = document.body.innerText;
        if (bodyText && bodyText.length > 50 && !processedTexts.has(bodyText.substring(0, 50))) {
             // Send first 2000 chars for analysis
             const textToSend = bodyText.substring(0, 2000);
             window.ReactNativeWebView.postMessage(JSON.stringify({
                type: 'text',
                content: textToSend
             }));
             processedTexts.add(bodyText.substring(0, 50)); // Cache by prefix
        }

        const images = document.getElementsByTagName('img');
        let processedCount = 0;
        for (let i = 0; i < images.length && processedCount < 10; i++) {
          const img = images[i];
          const src = img.src;
          
          if (src && !processedImages.has(src)) {
            if (src.startsWith('data:') || src.includes('image') || src.match(/\\.(jpg|jpeg|png|webp|gif|avif)/i)) {
              // Ignore small icons/tracking pixels (heuristic)
              if (img.width > 50 && img.height > 50) {
                  // Assign Unique ID for targetted blocking
                  if (!img.getAttribute('data-safebrowse-id')) {
                     img.setAttribute('data-safebrowse-id', 'sb-' + Math.random().toString(36).substr(2, 9));
                  }
                  const id = img.getAttribute('data-safebrowse-id');
                  
                  window.ReactNativeWebView.postMessage(JSON.stringify({
                    type: 'image',
                    content: src,
                    id: id
                  }));
                  processedImages.add(src);
                  processedCount++;
              }
            }
          }
        }
      }

      // Run less frequently to save resources
      setTimeout(scanContent, 2000);
      setInterval(scanContent, 5000); 
    })();
    true;
  `;

  return (
    <SafeAreaView style={styles.container} edges={['top']}>
      <StatusBar barStyle="light-content" />

      {/* Progress Bar */}
      {loadProgress > 0 && loadProgress < 100 && (
        <View style={styles.progressBarContainer}>
          <Animated.View
            style={[styles.progressBar, { width: `${loadProgress}%` }]}
          />
        </View>
      )}

      {/* Header (Top Bar with URL) */}
      <View style={styles.header}>
        <View style={styles.headerTopRow}>
          <View style={styles.profileSection}>
            <TouchableOpacity onPress={handleExitChildMode}>
              <View style={styles.profileBadge}>
                <View style={styles.avatar}>
                  <Text style={styles.avatarText}>{childName[0]}</Text>
                </View>
                <Animated.View
                  style={[
                    styles.safetyIndicator,
                    { transform: [{ scale: shieldScale }] }
                  ]}
                >
                  <Ionicons name="shield-checkmark" size={10} color="#fff" />
                </Animated.View>
              </View>
            </TouchableOpacity>
            <View style={styles.profileInfo}>
              <Text style={styles.profileName}>{childName}</Text>
            </View>
          </View>

          <View style={styles.actionButtons}>
            <TouchableOpacity style={styles.iconButton} onPress={addNewTab}>
              <Ionicons name="add" size={24} color="#f1f5f9" />
            </TouchableOpacity>
            <TouchableOpacity style={styles.iconButton} onPress={() => setTabsModalVisible(true)}>
              <Ionicons name="documents-outline" size={22} color="#f1f5f9" />
              <View style={styles.tabBadge}>
                <Text style={styles.tabBadgeText}>{tabs.length}</Text>
              </View>
            </TouchableOpacity>
          </View>
        </View>

      </View>

      {/* Main Content */}
      <View style={styles.mainContent}>
        {/* Shortcuts Section (Only if on Home/Google) */}
        {!blocked && currentUrl === 'https://www.google.com' && (
          <View style={styles.shortcutsSection}>
            <ScrollView
              ref={scrollRef}
              horizontal
              showsHorizontalScrollIndicator={false}
              contentContainerStyle={styles.shortcutsContainer}
            >
              {shortcuts.map((item, idx) => (
                <TouchableOpacity
                  key={idx}
                  style={styles.shortcut}
                  onPress={() => handleShortcutPress(item.url)}
                  activeOpacity={0.7}
                >
                  <View style={[styles.shortcutIcon, { backgroundColor: item.color + '20' }]}>
                    <Ionicons name={item.icon} size={28} color={item.color} />
                  </View>
                  <Text style={styles.shortcutLabel}>{item.label}</Text>
                </TouchableOpacity>
              ))}
            </ScrollView>
          </View>
        )}

        {/* Center Content Greetings (Only on Home) */}
        {!blocked && currentUrl === 'https://www.google.com' && (
          <View style={styles.centerContent}>
            <Text style={styles.greeting}>Hi {childName} 👋</Text>
            <Text style={styles.subGreeting}>What would you like to explore today?</Text>

            {/* Home Screen Search Bar */}
            <View style={styles.homeSearchContainer}>
              <View style={styles.searchBar}>
                <Ionicons name="lock-closed" size={14} color="#10b981" />
                <TextInput
                  style={styles.searchInput}
                  value={url}
                  onChangeText={(text) => {
                    setUrl(text);
                    setShowSuggestions(text.length > 0);
                  }}
                  onFocus={() => {
                    if (url === 'https://www.google.com') setUrl('');
                    setShowSuggestions(true);
                  }}
                  onSubmitEditing={handleNavigate}
                  placeholder="Search or enter URL"
                  placeholderTextColor="#64748b"
                  autoCapitalize="none"
                  autoCorrect={false}
                  selectTextOnFocus
                />
                {url && url !== 'https://www.google.com' ? (
                  <TouchableOpacity onPress={() => {
                    setUrl('');
                    setShowSuggestions(false);
                  }}>
                    <Ionicons name="close-circle" size={18} color="#64748b" />
                  </TouchableOpacity>
                ) : null}
              </View>

              {/* Suggestions Panel for Home */}
              {showSuggestions && (
                <View style={styles.suggestionsPanel}>
                  <Text style={styles.suggestionsTitle}>Safe Suggestions</Text>
                  {suggestions.map((item, idx) => (
                    <TouchableOpacity
                      key={idx}
                      style={styles.suggestionItem}
                      onPress={() => {
                        setUrl(item);
                        setShowSuggestions(false);
                        handleNavigate();
                      }}
                    >
                      <Ionicons name="search" size={16} color="#64748b" />
                      <Text style={styles.suggestionText}>{item}</Text>
                    </TouchableOpacity>
                  ))}
                </View>
              )}
            </View>

            {/* Stats (Moved here from old top bar) */}
            <View style={styles.statsBar}>
              <View style={styles.statBadge}>
                <Ionicons name="time-outline" size={14} color="#f59e0b" />
                <Text style={styles.statValue}>{screenTime}m</Text>
              </View>
              <View style={styles.statBadge}>
                <Ionicons name="shield-checkmark" size={14} color="#10b981" />
                <Text style={styles.statValue}>{blockedCount} Blocks</Text>
              </View>
            </View>
          </View>
        )}

        {/* WebView or Blocked Screen */}
        {blocked ? (
          <View style={styles.blockedContainer}>
            <Animated.View style={{ transform: [{ scale: pulseAnim }] }}>
              <Ionicons name="shield-outline" size={80} color="#ef4444" />
            </Animated.View>
            <Text style={styles.blockedTitle}>Content Blocked</Text>
            <Text style={styles.blockedReason}>{blockReason}</Text>
            <Text style={styles.blockedMessage}>
              This content has been blocked to keep you safe. If you think this is a mistake,
              ask your parent to review it.
            </Text>
            <TouchableOpacity
              style={styles.goBackButton}
              onPress={() => {
                setBlocked(false);
                setCurrentUrl('https://www.google.com');
              }}
            >
              <Text style={styles.goBackButtonText}>Go to Safe Home</Text>
            </TouchableOpacity>
          </View>
        ) : currentUrl !== 'https://www.google.com' ? (
          <WebView
            ref={webViewRef}
            source={{ uri: currentUrl }}
            style={styles.webview}
            onNavigationStateChange={handleWebViewNavigationStateChange}
            onMessage={handleWebViewMessage}
            injectedJavaScript={injectedJavaScript}
            onLoadProgress={({ nativeEvent }) => setLoadProgress(nativeEvent.progress * 100)}
            javaScriptEnabled
            domStorageEnabled
            startInLoadingState
          />
        ) : null}

        {/* Floating AI Shield */}
        {isScanning && (
          <Animated.View
            style={[
              styles.floatingShield,
              { transform: [{ scale: pulseAnim }] }
            ]}
          >
            <Ionicons name="shield-checkmark" size={20} color="#10b981" />
            <Text style={styles.floatingShieldText}>Scanning...</Text>
            <View style={styles.miniProgressBar}>
              <View style={[styles.miniProgressFill, { width: '70%' }]} />
            </View>
          </Animated.View>
        )}
      </View>

      {/* Bottom Search Bar (When Browsing) */}
      {!blocked && currentUrl !== 'https://www.google.com' && (
        <KeyboardAvoidingView
          behavior={Platform.OS === 'ios' ? 'padding' : undefined}
          keyboardVerticalOffset={Platform.OS === 'ios' ? 0 : 0}
        >
          <View style={styles.bottomBarContainer}>
            <View style={styles.searchBar}>
              <Ionicons name="lock-closed" size={14} color="#10b981" />
              <TextInput
                style={styles.searchInput}
                value={url}
                onChangeText={(text) => {
                  setUrl(text);
                }}
                onSubmitEditing={handleNavigate}
                placeholder="Search or enter URL"
                placeholderTextColor="#64748b"
                autoCapitalize="none"
                autoCorrect={false}
                selectTextOnFocus
              />
              <TouchableOpacity onPress={() => handleShortcutPress('https://www.google.com')}>
                <Ionicons name="home" size={20} color="#64748b" />
              </TouchableOpacity>
            </View>
          </View>
        </KeyboardAvoidingView>
      )}

      {/* Exit PIN Modal */}
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
              Enter your parent's PIN to exit Safe Browser
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

      {/* Tabs Modal */}
      <Modal visible={tabsModalVisible} animationType="slide" transparent>
        <View style={styles.modalContainer}>
          <View style={styles.tabsModal}>
            <View style={styles.modalHeader}>
              <Text style={styles.modalTitle}>Open Tabs ({tabs.length})</Text>
              <TouchableOpacity onPress={() => setTabsModalVisible(false)}>
                <Ionicons name="close" size={24} color="#94a3b8" />
              </TouchableOpacity>
            </View>
            <ScrollView>
              {tabs.map(tab => (
                <View key={tab.id} style={styles.tabItem}>
                  <TouchableOpacity
                    style={styles.tabItemContent}
                    onPress={() => {
                      setActiveTab(tab.id);
                      setCurrentUrl(tab.url);
                      setTabsModalVisible(false);
                    }}
                  >
                    <Text style={styles.tabEmoji}>{tab.favicon}</Text>
                    <Text style={styles.tabItemTitle}>{tab.title}</Text>
                  </TouchableOpacity>
                  <TouchableOpacity onPress={() => closeTab(tab.id)}>
                    <Ionicons name="close-circle" size={24} color="#64748b" />
                  </TouchableOpacity>
                </View>
              ))}
            </ScrollView>
          </View>
        </View>
      </Modal>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0f172a',
  },
  progressBarContainer: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    height: 3,
    backgroundColor: '#1e293b',
    zIndex: 100,
  },
  progressBar: {
    height: '100%',
    backgroundColor: '#10b981',
  },
  header: {
    backgroundColor: '#1e293b',
    borderBottomWidth: 1,
    borderBottomColor: '#334155',
    paddingBottom: 12,
  },
  headerTopRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingVertical: 12,
  },
  profileSection: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  profileBadge: {
    position: 'relative',
  },
  avatar: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: '#6366f1',
    alignItems: 'center',
    justifyContent: 'center',
  },
  avatarText: {
    fontSize: 14,
    fontWeight: '700',
    color: '#fff',
  },
  safetyIndicator: {
    position: 'absolute',
    bottom: -2,
    right: -2,
    width: 14,
    height: 14,
    borderRadius: 7,
    backgroundColor: '#10b981',
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 2,
    borderColor: '#1e293b',
  },
  profileInfo: {
    gap: 2,
  },
  profileName: {
    fontSize: 14,
    fontWeight: '600',
    color: '#f1f5f9',
  },
  actionButtons: {
    flexDirection: 'row',
    gap: 12,
  },
  iconButton: {
    padding: 6,
    position: 'relative'
  },
  tabBadge: {
    position: 'absolute',
    top: -4,
    right: -4,
    backgroundColor: '#6366f1',
    width: 16,
    height: 16,
    borderRadius: 8,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1.5,
    borderColor: '#1e293b'
  },
  tabBadgeText: {
    color: '#fff',
    fontSize: 9,
    fontWeight: 'bold'
  },
  // Removed old top searchBarContainer styles if they conflict, but repurposing:
  homeSearchContainer: {
    width: '100%',
    paddingHorizontal: 24,
    marginTop: 24,
    position: 'relative',
    zIndex: 50,
  },
  bottomBarContainer: {
    backgroundColor: '#1e293b',
    borderTopWidth: 1,
    borderTopColor: '#334155',
    paddingHorizontal: 16,
    paddingVertical: 12,
    paddingBottom: Platform.OS === 'ios' ? 24 : 12, // Safe area padding
  },
  searchBar: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#0f172a',
    borderRadius: 12,
    paddingHorizontal: 12,
    height: 44,
    gap: 10,
    borderWidth: 1,
    borderColor: '#334155',
  },
  searchInput: {
    flex: 1,
    color: '#f1f5f9',
    fontSize: 14,
  },
  suggestionsPanel: {
    marginTop: 8,
    backgroundColor: '#1e293b',
    borderRadius: 12,
    padding: 12,
    borderWidth: 1,
    borderColor: '#334155',
    gap: 8,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 4.65,
    elevation: 8,
  },
  suggestionsTitle: {
    fontSize: 12,
    color: '#64748b',
    fontWeight: '600',
    marginBottom: 4,
  },
  suggestionItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    padding: 12,
    backgroundColor: '#0f172a',
    borderRadius: 8,
  },
  suggestionText: {
    fontSize: 14,
    color: '#f1f5f9',
  },
  statsBar: {
    flexDirection: 'row',
    gap: 8,
    marginTop: 16
  },
  statBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    backgroundColor: '#334155',
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 12,
  },
  statValue: {
    fontSize: 12,
    fontWeight: '700',
    color: '#f1f5f9',
  },
  mainContent: {
    flex: 1,
  },
  shortcutsSection: {
    paddingVertical: 20,
    backgroundColor: '#1e293b',
    borderBottomWidth: 1,
    borderBottomColor: '#334155',
  },
  shortcutsContainer: {
    paddingHorizontal: 16,
    gap: 16,
  },
  shortcut: {
    alignItems: 'center',
    gap: 8,
    marginRight: 8,
  },
  shortcutIcon: {
    width: 64,
    height: 64,
    borderRadius: 20,
    alignItems: 'center',
    justifyContent: 'center',
  },
  shortcutLabel: {
    fontSize: 12,
    color: '#94a3b8',
    fontWeight: '500',
  },
  centerContent: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingBottom: 100,
  },
  greeting: {
    fontSize: 28,
    fontWeight: '700',
    color: '#f1f5f9',
    marginBottom: 8,
  },
  subGreeting: {
    fontSize: 16,
    color: '#94a3b8',
  },
  webview: {
    flex: 1,
    backgroundColor: '#fff',
  },
  blockedContainer: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 24,
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
  goBackButton: {
    backgroundColor: '#6366f1',
    paddingHorizontal: 32,
    paddingVertical: 16,
    borderRadius: 12,
    marginTop: 32,
  },
  goBackButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  floatingShield: {
    position: 'absolute',
    top: 16,
    right: 16,
    backgroundColor: '#1e293b',
    borderRadius: 12,
    padding: 12,
    borderWidth: 2,
    borderColor: '#10b981',
    gap: 6,
    minWidth: 140,
  },
  floatingShieldText: {
    color: '#10b981',
    fontSize: 14,
    fontWeight: '600',
  },
  miniProgressBar: {
    height: 4,
    backgroundColor: '#334155',
    borderRadius: 2,
    overflow: 'hidden',
  },
  miniProgressFill: {
    height: '100%',
    backgroundColor: '#10b981',
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
  tabsModal: {
    backgroundColor: '#1e293b',
    borderRadius: 24,
    padding: 24,
    maxHeight: '80%',
  },
  tabItem: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#334155',
  },
  tabItemContent: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    flex: 1,
  },
  tabEmoji: {
    fontSize: 24,
  },
  tabItemTitle: {
    fontSize: 16,
    color: '#f1f5f9',
    fontWeight: '500',
  },
});

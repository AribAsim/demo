import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  RefreshControl,
  Alert,
} from 'react-native';
import { useRouter } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useAuth } from '../../contexts/AuthContext';
import { useAppMode } from '../../contexts/AppModeContext';
import axios from 'axios';
import Constants from 'expo-constants';
import WellbeingSection from '../../components/WellbeingSection';

const API_URL = Constants.expoConfig?.extra?.EXPO_PUBLIC_BACKEND_URL || process.env.EXPO_PUBLIC_BACKEND_URL;

interface Profile {
  id: string;
  name: string;
  age: number;
  maturity_level: string;
}

interface Log {
  id: string;
  profile_name: string;
  content_type: string;
  detected_at: string;
  reasons: string[];
}

export default function Dashboard() {
  const { user, token } = useAuth();
  const { setMode, setSelectedProfile } = useAppMode();
  const router = useRouter();
  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [recentLogs, setRecentLogs] = useState<Log[]>([]);
  const [refreshing, setRefreshing] = useState(false);

  const [loading, setLoading] = useState(true);
  const [selectedStatsProfile, setSelectedStatsProfile] = useState<string | null>(null);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      const [profilesRes, logsRes] = await Promise.all([
        axios.get(`${API_URL}/api/profiles`, {
          headers: { Authorization: `Bearer ${token}` },
        }),
        axios.get(`${API_URL}/api/logs?limit=5`, {
          headers: { Authorization: `Bearer ${token}` },
        }),
      ]);

      setProfiles(profilesRes.data);
      if (profilesRes.data.length > 0 && !selectedStatsProfile) {
        setSelectedStatsProfile(profilesRes.data[0].id);
      }
      setRecentLogs(logsRes.data);
    } catch (error: any) {
      console.error('Error loading data:', error);
      Alert.alert('Error', 'Failed to load data');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  const handleRefresh = () => {
    setRefreshing(true);
    loadData();
  };

  const enterChildMode = (profile: Profile) => {
    if (!user?.pin) {
      Alert.alert(
        'PIN Required',
        'Please set up a PIN in Settings before entering Child Mode',
        [
          { text: 'Cancel', style: 'cancel' },
          { text: 'Go to Settings', onPress: () => router.push('/parent/settings') },
        ]
      );
      return;
    }

    setSelectedProfile(profile.id);
    setMode('child');
    router.push('/child/browser');
  };

  const getMaturityColor = (level: string) => {
    switch (level) {
      case 'strict':
        return '#ef4444';
      case 'moderate':
        return '#f59e0b';
      case 'lenient':
        return '#10b981';
      default:
        return '#6366f1';
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView
        style={styles.scrollView}
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={handleRefresh} />}
      >
        <View style={styles.header}>
          <View>
            <Text style={styles.greeting}>Welcome back,</Text>
            <Text style={styles.name}>{user?.name}</Text>
          </View>
          <Ionicons name="shield-checkmark" size={40} color="#6366f1" />
        </View>

        <View style={styles.section}>
          <View style={styles.sectionHeader}>
            <Text style={styles.sectionTitle}>Child Profiles</Text>
            <TouchableOpacity onPress={() => router.push('/parent/profiles')}>
              <Text style={styles.seeAllText}>See All</Text>
            </TouchableOpacity>
          </View>

          {profiles.length === 0 ? (
            <View style={styles.emptyCard}>
              <Ionicons name="person-add" size={48} color="#64748b" />
              <Text style={styles.emptyText}>No profiles yet</Text>
              <TouchableOpacity
                style={styles.emptyButton}
                onPress={() => router.push('/parent/profiles')}
              >
                <Text style={styles.emptyButtonText}>Create Profile</Text>
              </TouchableOpacity>
            </View>
          ) : (
            profiles.slice(0, 3).map((profile) => (
              <TouchableOpacity
                key={profile.id}
                style={styles.profileCard}
                onPress={() => enterChildMode(profile)}
              >
                <View style={styles.profileIcon}>
                  <Ionicons name="person" size={24} color="#6366f1" />
                </View>
                <View style={styles.profileInfo}>
                  <Text style={styles.profileName}>{profile.name}</Text>
                  <Text style={styles.profileDetails}>{profile.age} years old</Text>
                </View>
                <View
                  style={[
                    styles.maturityBadge,
                    { backgroundColor: getMaturityColor(profile.maturity_level) + '20' },
                  ]}
                >
                  <Text
                    style={[
                      styles.maturityText,
                      { color: getMaturityColor(profile.maturity_level) },
                    ]}
                  >
                    {profile.maturity_level}
                  </Text>
                </View>
                <Ionicons name="chevron-forward" size={20} color="#64748b" />
              </TouchableOpacity>
            ))
          )}
        </View>

        {profiles.length > 0 && selectedStatsProfile && (
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Digital Wellbeing</Text>
            <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.tabContainer}>
              {profiles.map(profile => (
                <TouchableOpacity
                  key={profile.id}
                  style={[
                    styles.tab,
                    selectedStatsProfile === profile.id && styles.activeTab
                  ]}
                  onPress={() => setSelectedStatsProfile(profile.id)}
                >
                  <Text style={[
                    styles.tabText,
                    selectedStatsProfile === profile.id && styles.activeTabText
                  ]}>
                    {profile.name}
                  </Text>
                </TouchableOpacity>
              ))}
            </ScrollView>
            <WellbeingSection profileId={selectedStatsProfile} />
          </View>
        )}

        <View style={styles.section}>
          <View style={styles.sectionHeader}>
            <Text style={styles.sectionTitle}>Recent Activity</Text>
            <TouchableOpacity onPress={() => router.push('/parent/logs')}>
              <Text style={styles.seeAllText}>See All</Text>
            </TouchableOpacity>
          </View>

          {recentLogs.length === 0 ? (
            <View style={styles.emptyCard}>
              <Ionicons name="checkmark-circle" size={48} color="#10b981" />
              <Text style={styles.emptyText}>No harmful content detected</Text>
              <Text style={styles.emptySubtext}>Your children are browsing safely</Text>
            </View>
          ) : (
            recentLogs.map((log) => (
              <View key={log.id} style={styles.logCard}>
                <View style={styles.logIcon}>
                  <Ionicons name="alert-circle" size={24} color="#ef4444" />
                </View>
                <View style={styles.logInfo}>
                  <Text style={styles.logProfile}>{log.profile_name}</Text>
                  <Text style={styles.logReason}>{log.reasons[0]}</Text>
                  <Text style={styles.logTime}>
                    {new Date(log.detected_at).toLocaleString()}
                  </Text>
                </View>
              </View>
            ))
          )}
        </View>

        <View style={styles.statsCard}>
          <Text style={styles.statsTitle}>Protection Stats</Text>
          <View style={styles.statsGrid}>
            <View style={styles.statItem}>
              <Text style={styles.statValue}>{profiles.length}</Text>
              <Text style={styles.statLabel}>Profiles</Text>
            </View>
            <View style={styles.statItem}>
              <Text style={styles.statValue}>{recentLogs.length}</Text>
              <Text style={styles.statLabel}>Alerts Today</Text>
            </View>
            <View style={styles.statItem}>
              <Text style={[styles.statValue, { color: '#10b981' }]}>Active</Text>
              <Text style={styles.statLabel}>Status</Text>
            </View>
          </View>
        </View>
      </ScrollView>
    </SafeAreaView >
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0f172a',
  },
  scrollView: {
    flex: 1,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 24,
  },
  greeting: {
    fontSize: 16,
    color: '#94a3b8',
  },
  name: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#f1f5f9',
    marginTop: 4,
  },
  section: {
    padding: 24,
    paddingTop: 0,
  },
  sectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: '600',
    color: '#f1f5f9',
  },
  seeAllText: {
    color: '#6366f1',
    fontSize: 14,
  },
  profileCard: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#1e293b',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: '#334155',
  },
  profileIcon: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: '#6366f120',
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 12,
  },
  profileInfo: {
    flex: 1,
  },
  profileName: {
    fontSize: 18,
    fontWeight: '600',
    color: '#f1f5f9',
  },
  profileDetails: {
    fontSize: 14,
    color: '#94a3b8',
    marginTop: 2,
  },
  maturityBadge: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 12,
    marginRight: 8,
  },
  maturityText: {
    fontSize: 12,
    fontWeight: '600',
    textTransform: 'capitalize',
  },
  logCard: {
    flexDirection: 'row',
    backgroundColor: '#1e293b',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: '#334155',
  },
  logIcon: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: '#ef444420',
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 12,
  },
  logInfo: {
    flex: 1,
  },
  logProfile: {
    fontSize: 16,
    fontWeight: '600',
    color: '#f1f5f9',
  },
  logReason: {
    fontSize: 14,
    color: '#94a3b8',
    marginTop: 2,
  },
  logTime: {
    fontSize: 12,
    color: '#64748b',
    marginTop: 4,
  },
  emptyCard: {
    backgroundColor: '#1e293b',
    borderRadius: 12,
    padding: 32,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#334155',
  },
  emptyText: {
    fontSize: 16,
    color: '#94a3b8',
    marginTop: 16,
  },
  emptySubtext: {
    fontSize: 14,
    color: '#64748b',
    marginTop: 8,
  },
  emptyButton: {
    backgroundColor: '#6366f1',
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 8,
    marginTop: 16,
  },
  emptyButtonText: {
    color: '#ffffff',
    fontSize: 16,
    fontWeight: '600',
  },
  statsCard: {
    backgroundColor: '#1e293b',
    borderRadius: 12,
    padding: 24,
    margin: 24,
    marginTop: 0,
    borderWidth: 1,
    borderColor: '#334155',
  },
  statsTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#f1f5f9',
    marginBottom: 16,
  },
  statsGrid: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  statItem: {
    alignItems: 'center',
  },
  statValue: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#6366f1',
  },
  statLabel: {
    fontSize: 12,
    color: '#94a3b8',
    marginTop: 4,
  },
  tabContainer: {
    flexDirection: 'row',
    marginBottom: 16,
  },
  tab: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
    backgroundColor: '#334155',
    marginRight: 8,
  },
  activeTab: {
    backgroundColor: '#6366f1',
  },
  tabText: {
    color: '#94a3b8',
    fontWeight: '600',
  },
  activeTabText: {
    color: '#ffffff',
  }
});

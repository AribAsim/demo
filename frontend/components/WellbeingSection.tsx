import React, { useState, useCallback } from 'react';
import { View, Text, StyleSheet, Dimensions, ActivityIndicator } from 'react-native';
import { useFocusEffect } from '@react-navigation/native';
import { LineChart, BarChart } from 'react-native-chart-kit';
import { Ionicons } from '@expo/vector-icons';
import axios from 'axios';
import Constants from 'expo-constants';
import { useAuth } from '../contexts/AuthContext';

const API_URL = Constants.expoConfig?.extra?.EXPO_PUBLIC_BACKEND_URL || process.env.EXPO_PUBLIC_BACKEND_URL;
const SCREEN_WIDTH = Dimensions.get('window').width;

interface DailyStat {
    date: string;
    screen_time_minutes: number;
    unsafe_count: number;
}

interface WellbeingData {
    total_screen_time_minutes: number;
    avg_daily_minutes: number;
    daily_stats: DailyStat[];
    unsafe_detections_total: number;
}

export default function WellbeingSection({ profileId }: { profileId: string }) {
    const { token } = useAuth();
    const [data, setData] = useState<WellbeingData | null>(null);
    const [loading, setLoading] = useState(true);

    const fetchWellbeingData = useCallback(async () => {
        try {
            const response = await axios.get(
                `${API_URL}/api/parent/digital-wellbeing/${profileId}`,
                { headers: { Authorization: `Bearer ${token}` } }
            );
            setData(response.data);
        } catch (error) {
            console.error('Failed to fetch wellbeing data:', error);
        } finally {
            setLoading(false);
        }
    }, [profileId, token]);

    useFocusEffect(
        useCallback(() => {
            fetchWellbeingData();

            const interval = setInterval(() => {
                fetchWellbeingData();
            }, 30000); // Poll every 30 seconds

            return () => clearInterval(interval);
        }, [fetchWellbeingData])
    );

    if (loading) {
        return (
            <View style={styles.loadingContainer}>
                <ActivityIndicator size="small" color="#6366f1" />
            </View>
        );
    }

    if (!data) return null;

    // Prepare Chart Data
    const screenTimeLabels = data.daily_stats.map(d => d.date.slice(5)); // 'MM-DD'
    const screenTimeValues = data.daily_stats.map(d => d.screen_time_minutes);
    const unsafeValues = data.daily_stats.map(d => d.unsafe_count);

    return (
        <View style={styles.container}>
            <Text style={styles.sectionTitle}>Digital Wellbeing</Text>

            {/* Summary Cards */}
            <View style={styles.summaryRow}>
                <View style={[styles.card, { backgroundColor: '#3b82f6' }]}>
                    <Ionicons name="time-outline" size={24} color="#fff" />
                    <Text style={styles.cardValue}>{Math.round(data.total_screen_time_minutes / 60)}h {data.total_screen_time_minutes % 60}m</Text>
                    <Text style={styles.cardLabel}>Est. Screen Time (7d)</Text>
                </View>

                <View style={[styles.card, { backgroundColor: '#ef4444' }]}>
                    <Ionicons name="alert-circle-outline" size={24} color="#fff" />
                    <Text style={styles.cardValue}>{data.unsafe_detections_total}</Text>
                    <Text style={styles.cardLabel}>Unsafe Detections</Text>
                </View>
            </View>

            {/* Screen Time Chart */}
            <View style={styles.chartContainer}>
                <Text style={styles.chartTitle}>Daily Screen Time (Minutes)</Text>
                <LineChart
                    data={{
                        labels: screenTimeLabels,
                        datasets: [{ data: screenTimeValues }]
                    }}
                    width={SCREEN_WIDTH - 48} // Padding adjustments
                    height={220}
                    yAxisLabel=""
                    yAxisSuffix="m"
                    chartConfig={{
                        backgroundColor: '#1e293b',
                        backgroundGradientFrom: '#1e293b',
                        backgroundGradientTo: '#1e293b',
                        decimalPlaces: 0,
                        color: (opacity = 1) => `rgba(99, 102, 241, ${opacity})`,
                        labelColor: (opacity = 1) => `rgba(148, 163, 184, ${opacity})`,
                        style: { borderRadius: 16 },
                        propsForDots: { r: '4', strokeWidth: '2', stroke: '#6366f1' }
                    }}
                    bezier
                    style={{ borderRadius: 16, marginVertical: 8 }}
                />
                <Text style={styles.disclaimer}>* Screen time is an estimate based on active browsing sessions.</Text>
            </View>
        </View>
    );
}

const styles = StyleSheet.create({
    container: {
        marginVertical: 16,
    },
    loadingContainer: {
        padding: 20,
        alignItems: 'center',
    },
    sectionTitle: {
        fontSize: 20,
        fontWeight: 'bold',
        color: '#f1f5f9',
        marginBottom: 16,
    },
    summaryRow: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        marginBottom: 24,
    },
    card: {
        flex: 1,
        padding: 16,
        borderRadius: 16,
        marginHorizontal: 4,
        alignItems: 'center',
    },
    cardValue: {
        fontSize: 24,
        fontWeight: 'bold',
        color: '#fff',
        marginTop: 8,
    },
    cardLabel: {
        fontSize: 12,
        color: 'rgba(255,255,255,0.8)',
        marginTop: 4,
    },
    chartContainer: {
        backgroundColor: '#1e293b',
        borderRadius: 16,
        padding: 16,
        alignItems: 'center',
    },
    chartTitle: {
        fontSize: 16,
        fontWeight: '600',
        color: '#cbd5e1',
        marginBottom: 12,
        alignSelf: 'flex-start',
    },
    disclaimer: {
        fontSize: 10,
        color: '#64748b',
        marginTop: 8,
        alignSelf: 'flex-start',
    }
});

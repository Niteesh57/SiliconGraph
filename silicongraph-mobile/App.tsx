import React, { useState } from 'react';
import {
  SafeAreaView,
  StatusBar,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
  ScrollView,
  useColorScheme,
} from 'react-native';
import {
  SafeAreaProvider,
} from 'react-native-safe-area-context';

function App(): React.JSX.Element {
  const isDarkMode = useColorScheme() === 'dark';
  const [nodeCount, setNodeCount] = useState(128);
  const [edgeCount, setEdgeCount] = useState(512);

  const handleAddNodes = () => {
    setNodeCount(prev => prev + 16);
    setEdgeCount(prev => prev + 64);
  };

  const handleReset = () => {
    setNodeCount(128);
    setEdgeCount(512);
  };

  return (
    <SafeAreaProvider>
      <StatusBar barStyle={isDarkMode ? 'light-content' : 'dark-content'} />
      <SafeAreaView style={[styles.container, isDarkMode ? styles.darkBg : styles.lightBg]}>
        <ScrollView contentContainerStyle={styles.scrollContent}>
          
          {/* Header */}
          <View style={styles.header}>
            <View style={styles.badge}>
              <View style={styles.badgeDot} />
              <Text style={styles.badgeText}>METRO BUNDLER ACTIVE</Text>
            </View>
            <Text style={[styles.title, isDarkMode ? styles.darkText : styles.lightText]}>
              SiliconGraph Mobile
            </Text>
            <Text style={styles.subtitle}>
              React Native 0.86 High-Performance Mobile Graph Engine
            </Text>
          </View>

          {/* Metrics Cards */}
          <View style={styles.grid}>
            <View style={[styles.card, isDarkMode ? styles.darkCard : styles.lightCard]}>
              <Text style={styles.cardLabel}>GRAPH NODES</Text>
              <Text style={styles.cardValue}>{nodeCount}</Text>
              <Text style={styles.cardSubtext}>Active in VRAM</Text>
            </View>

            <View style={[styles.card, isDarkMode ? styles.darkCard : styles.lightCard]}>
              <Text style={styles.cardLabel}>GRAPH EDGES</Text>
              <Text style={styles.cardValue}>{edgeCount}</Text>
              <Text style={styles.cardSubtext}>Connected Topology</Text>
            </View>
          </View>

          {/* Status Panel */}
          <View style={[styles.panel, isDarkMode ? styles.darkCard : styles.lightCard]}>
            <Text style={[styles.panelTitle, isDarkMode ? styles.darkText : styles.lightText]}>
              System Status
            </Text>
            <View style={styles.statusRow}>
              <Text style={styles.statusKey}>Bundler Engine:</Text>
              <Text style={styles.statusVal}>Metro v0.86.0</Text>
            </View>
            <View style={styles.statusRow}>
              <Text style={styles.statusKey}>Architecture:</Text>
              <Text style={styles.statusVal}>React Native Fabric</Text>
            </View>
            <View style={styles.statusRow}>
              <Text style={styles.statusKey}>Compute Mode:</Text>
              <Text style={styles.statusVal}>Hardware Accelerated</Text>
            </View>
          </View>

          {/* Interactive Controls */}
          <View style={styles.actions}>
            <TouchableOpacity style={styles.primaryButton} onPress={handleAddNodes}>
              <Text style={styles.primaryButtonText}>+ Add 16 Graph Nodes</Text>
            </TouchableOpacity>

            <TouchableOpacity style={styles.secondaryButton} onPress={handleReset}>
              <Text style={styles.secondaryButtonText}>Reset Topology</Text>
            </TouchableOpacity>
          </View>

        </ScrollView>
      </SafeAreaView>
    </SafeAreaProvider>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  lightBg: {
    backgroundColor: '#F8FAFC',
  },
  darkBg: {
    backgroundColor: '#0F172A',
  },
  scrollContent: {
    padding: 24,
  },
  header: {
    marginBottom: 24,
  },
  badge: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(16, 185, 129, 0.15)',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 20,
    alignSelf: 'flex-start',
    marginBottom: 12,
  },
  badgeDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: '#10B981',
    marginRight: 8,
  },
  badgeText: {
    color: '#10B981',
    fontSize: 11,
    fontWeight: '700',
    letterSpacing: 0.8,
  },
  title: {
    fontSize: 28,
    fontWeight: '800',
    letterSpacing: -0.5,
    marginBottom: 6,
  },
  subtitle: {
    fontSize: 14,
    color: '#64748B',
    lineHeight: 20,
  },
  lightText: {
    color: '#0F172A',
  },
  darkText: {
    color: '#F8FAFC',
  },
  grid: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 20,
  },
  card: {
    flex: 1,
    padding: 16,
    borderRadius: 16,
    borderWidth: 1,
  },
  lightCard: {
    backgroundColor: '#FFFFFF',
    borderColor: '#E2E8F0',
  },
  darkCard: {
    backgroundColor: '#1E293B',
    borderColor: '#334155',
  },
  cardLabel: {
    fontSize: 11,
    fontWeight: '700',
    color: '#64748B',
    letterSpacing: 0.5,
    marginBottom: 8,
  },
  cardValue: {
    fontSize: 32,
    fontWeight: '800',
    color: '#3B82F6',
    marginBottom: 4,
  },
  cardSubtext: {
    fontSize: 12,
    color: '#94A3B8',
  },
  panel: {
    padding: 20,
    borderRadius: 16,
    borderWidth: 1,
    marginBottom: 24,
  },
  panelTitle: {
    fontSize: 16,
    fontWeight: '700',
    marginBottom: 16,
  },
  statusRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(148, 163, 184, 0.1)',
  },
  statusKey: {
    fontSize: 13,
    color: '#64748B',
  },
  statusVal: {
    fontSize: 13,
    fontWeight: '600',
    color: '#3B82F6',
  },
  actions: {
    gap: 12,
  },
  primaryButton: {
    backgroundColor: '#2563EB',
    paddingVertical: 16,
    borderRadius: 14,
    alignItems: 'center',
    shadowColor: '#2563EB',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 4,
  },
  primaryButtonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '700',
  },
  secondaryButton: {
    backgroundColor: 'transparent',
    paddingVertical: 14,
    borderRadius: 14,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#475569',
  },
  secondaryButtonText: {
    color: '#94A3B8',
    fontSize: 14,
    fontWeight: '600',
  },
});

export default App;


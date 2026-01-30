import React from 'react';
import { StatusBar } from 'expo-status-bar';
import { NavigationContainer, DefaultTheme } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { Ionicons } from '@expo/vector-icons';
import { Platform } from 'react-native';

import HomeScreen from './src/screens/HomeScreen';
import ScanScreen from './src/screens/ScanScreen';
import HistoryScreen from './src/screens/HistoryScreen';
import ProposalScreen from './src/screens/ProposalScreen';
import AboutUsScreen from './src/screens/AboutUsScreen';
import PublicationsScreen from './src/screens/PublicationsScreen';
import DirectoryScreen from './src/screens/DirectoryScreen';
import NewsScreen from './src/screens/NewsScreen';
import EventsScreen from './src/screens/EventsScreen';
import LoginScreen from './src/screens/LoginScreen';
import RegisterScreen from './src/screens/RegisterScreen';
import AccountScreen from './src/screens/AccountScreen';
import AdminScreen from './src/screens/AdminScreen';
import { AuthProvider, AuthContext } from './src/context/AuthContext';

const Tab = createBottomTabNavigator();
const Stack = createNativeStackNavigator();

const navTheme = {
  ...DefaultTheme,
  colors: {
    ...DefaultTheme.colors,
    background: '#eef1f3',
    card: '#ffffff',
    text: '#1f2937',
    border: '#d7dde3',
    primary: '#1f7a4f',
  },
};

function AppTabs() {
  return (
    <Tab.Navigator
      screenOptions={({ route }) => ({
        headerShown: false,
        tabBarStyle: Platform.OS === 'web' ? { display: 'none' } : { backgroundColor: '#ffffff', borderTopColor: '#d7dde3' },
        tabBarActiveTintColor: '#1f7a4f',
        tabBarInactiveTintColor: 'rgba(31,41,55,0.6)',
        tabBarIcon: ({ color, size }) => {
          const map = {
            Home: 'home',
            Scan: 'camera',
            History: 'time',
            Proposal: 'document-text',
            Account: 'person',
          };
          const name = map[route.name] ?? 'ellipse';
          return <Ionicons name={name} size={size} color={color} />;
        },
      })}
    >
      <Tab.Screen name="Home" component={HomeScreen} />
      <Tab.Screen name="Scan" component={ScanScreen} />
      <Tab.Screen name="History" component={HistoryScreen} />
      <Tab.Screen name="Proposal" component={ProposalScreen} />
      <Tab.Screen name="Account" component={AccountScreen} />
    </Tab.Navigator>
  );
}

function RootNavigator() {
  return (
    <Stack.Navigator screenOptions={{ headerShown: false }}>
      <Stack.Screen name="Tabs" component={AppTabs} />
      <Stack.Screen name="AboutUs" component={AboutUsScreen} />
      <Stack.Screen name="Publications" component={PublicationsScreen} />
      <Stack.Screen name="Directory" component={DirectoryScreen} />
      <Stack.Screen name="News" component={NewsScreen} />
      <Stack.Screen name="Events" component={EventsScreen} />
      <Stack.Screen name="Login" component={LoginScreen} />
      <Stack.Screen name="Register" component={RegisterScreen} />
      <Stack.Screen name="Admin" component={AdminScreen} />
    </Stack.Navigator>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <NavigationContainer theme={navTheme}>
        <StatusBar style={Platform.OS === 'web' ? 'dark' : 'dark'} />
        <RootNavigator />
      </NavigationContainer>
    </AuthProvider>
  );
}

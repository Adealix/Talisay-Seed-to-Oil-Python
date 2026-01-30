import React, { useMemo, useState } from 'react';
import { Platform, Pressable, StyleSheet, Text, TextInput, View } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import Screen from '../components/Screen';
import { Body, Card, Title, Kicker } from '../components/Ui';
import { useAuth } from '../hooks';
import { theme } from '../theme/theme';

export default function LoginScreen({ navigation }) {
  const { login } = useAuth();

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [showPassword, setShowPassword] = useState(false);

  const canSubmit = useMemo(() => email.trim().length > 3 && password.length >= 6 && !busy, [email, password, busy]);

  async function onLogin() {
    setError(null);
    setBusy(true);
    try {
      await login({ email: email.trim(), password });
      // Navigate back to Account screen after successful login
      navigation.navigate('Account');
    } catch (e) {
      setError(e?.message || 'login_failed');
    } finally {
      setBusy(false);
    }
  }

  return (
    <Screen>
      {/* Header */}
      <View style={styles.pageHeader}>
        <View style={styles.headerIcon}>
          <Ionicons name="log-in" size={40} color="#fff" />
        </View>
        <View>
          <Title style={styles.headerTitle}>Welcome Back</Title>
          <Body style={styles.headerSubtitle}>Sign in to your account</Body>
        </View>
      </View>

      {/* Login Form */}
      <Card>
        <Kicker>Login Credentials</Kicker>

        {/* Email Field */}
        <View style={styles.fieldWrapper}>
          <View style={styles.fieldIcon}>
            <Ionicons name="mail" size={20} color={theme.colors.green} />
          </View>
          <TextInput
            value={email}
            onChangeText={setEmail}
            autoCapitalize="none"
            keyboardType={Platform.OS === 'web' ? 'default' : 'email-address'}
            placeholder="Email address"
            placeholderTextColor="#999"
            style={styles.fieldInput}
          />
        </View>

        {/* Password Field */}
        <View style={styles.fieldWrapper}>
          <View style={styles.fieldIcon}>
            <Ionicons name="lock-closed" size={20} color={theme.colors.green} />
          </View>
          <TextInput
            value={password}
            onChangeText={setPassword}
            secureTextEntry={!showPassword}
            placeholder="Password"
            placeholderTextColor="#999"
            style={styles.fieldInput}
          />
          <Pressable 
            style={styles.eyeBtn} 
            onPress={() => setShowPassword(!showPassword)}
          >
            <Ionicons 
              name={showPassword ? 'eye-off' : 'eye'} 
              size={20} 
              color={theme.colors.muted} 
            />
          </Pressable>
        </View>

        {/* Error Message */}
        {error && (
          <View style={styles.errorBox}>
            <Ionicons name="alert-circle" size={18} color={theme.colors.danger} />
            <Text style={styles.errorText}>{error}</Text>
          </View>
        )}

        {/* Submit Button */}
        <Pressable 
          style={[styles.submitBtn, !canSubmit && styles.submitBtnDisabled]} 
          onPress={onLogin}
          disabled={!canSubmit}
        >
          {busy ? (
            <Ionicons name="hourglass" size={20} color="#fff" />
          ) : (
            <Ionicons name="log-in" size={20} color="#fff" />
          )}
          <Text style={styles.submitBtnText}>
            {busy ? 'Signing in...' : 'Login'}
          </Text>
        </Pressable>

        {/* Divider */}
        <View style={styles.dividerRow}>
          <View style={styles.dividerLine} />
          <Text style={styles.dividerText}>OR</Text>
          <View style={styles.dividerLine} />
        </View>

        {/* Register Link */}
        <Pressable 
          style={styles.secondaryBtn} 
          onPress={() => navigation.navigate('Register')}
        >
          <Ionicons name="person-add" size={20} color={theme.colors.green} />
          <Text style={styles.secondaryBtnText}>Create New Account</Text>
        </Pressable>
      </Card>

      {/* Note Card */}
      <Card style={styles.noteCard}>
        <View style={styles.noteRow}>
          <Ionicons name="information-circle" size={20} color={theme.colors.green} />
          <Body style={styles.noteText}>
            This is a prototype. Use a simple password, but don't reuse a real one.
          </Body>
        </View>
      </Card>
    </Screen>
  );
}

const styles = StyleSheet.create({
  /* Page Header */
  pageHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 16,
    marginBottom: 20,
    padding: 24,
    backgroundColor: theme.colors.greenDark,
    borderRadius: 12,
  },
  headerIcon: {
    width: 72,
    height: 72,
    borderRadius: 36,
    backgroundColor: 'rgba(255,255,255,0.2)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  headerTitle: {
    color: '#ffffff',
    marginBottom: 0,
  },
  headerSubtitle: {
    color: 'rgba(255,255,255,0.8)',
    marginTop: 4,
  },

  /* Field Wrapper */
  fieldWrapper: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#f9f9f9',
    borderRadius: 10,
    borderWidth: 1,
    borderColor: theme.colors.border,
    marginBottom: 12,
    overflow: 'hidden',
  },
  fieldIcon: {
    width: 48,
    height: 48,
    backgroundColor: '#e8f5e9',
    alignItems: 'center',
    justifyContent: 'center',
  },
  fieldInput: {
    flex: 1,
    paddingHorizontal: 14,
    paddingVertical: 14,
    fontSize: 15,
    color: theme.colors.text,
  },
  eyeBtn: {
    width: 48,
    height: 48,
    alignItems: 'center',
    justifyContent: 'center',
  },

  /* Error */
  errorBox: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    backgroundColor: '#ffebee',
    padding: 12,
    borderRadius: 8,
    marginBottom: 12,
  },
  errorText: {
    color: theme.colors.danger,
    fontWeight: '600',
    flex: 1,
  },

  /* Submit Button */
  submitBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 10,
    backgroundColor: theme.colors.green,
    paddingVertical: 16,
    borderRadius: 10,
    marginTop: 4,
  },
  submitBtnDisabled: {
    opacity: 0.5,
  },
  submitBtnText: {
    color: '#fff',
    fontWeight: '800',
    fontSize: 16,
  },

  /* Divider */
  dividerRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginVertical: 20,
  },
  dividerLine: {
    flex: 1,
    height: 1,
    backgroundColor: theme.colors.border,
  },
  dividerText: {
    paddingHorizontal: 16,
    color: theme.colors.muted,
    fontSize: 12,
    fontWeight: '700',
  },

  /* Secondary Button */
  secondaryBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 10,
    backgroundColor: '#e8f5e9',
    paddingVertical: 14,
    borderRadius: 10,
  },
  secondaryBtnText: {
    color: theme.colors.green,
    fontWeight: '700',
    fontSize: 15,
  },

  /* Note Card */
  noteCard: {
    backgroundColor: '#e8f5e9',
    borderColor: theme.colors.green,
  },
  noteRow: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 12,
  },
  noteText: {
    flex: 1,
    color: theme.colors.greenDark,
  },
});

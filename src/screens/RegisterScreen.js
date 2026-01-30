import React, { useMemo, useState } from 'react';
import { Platform, Pressable, StyleSheet, Text, TextInput, View } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import Screen from '../components/Screen';
import { Body, Card, Title, Kicker, Label } from '../components/Ui';
import { useAuth } from '../hooks';
import { theme } from '../theme/theme';

export default function RegisterScreen({ navigation }) {
  const { register } = useAuth();

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [role, setRole] = useState('user');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [showPassword, setShowPassword] = useState(false);

  const canSubmit = useMemo(() => 
    email.trim().length > 3 && 
    password.length >= 6 && 
    password === confirmPassword &&
    !busy, 
    [email, password, confirmPassword, busy]
  );

  async function onRegister() {
    setError(null);
    setBusy(true);
    try {
      await register({ email: email.trim(), password, role });
      // Navigate back to Account screen after successful registration
      navigation.navigate('Account');
    } catch (e) {
      setError(e?.message || 'register_failed');
    } finally {
      setBusy(false);
    }
  }

  return (
    <Screen>
      {/* Header */}
      <View style={styles.pageHeader}>
        <View style={styles.headerIcon}>
          <Ionicons name="person-add" size={40} color="#fff" />
        </View>
        <View>
          <Title style={styles.headerTitle}>Create Account</Title>
          <Body style={styles.headerSubtitle}>Join us to save your predictions</Body>
        </View>
      </View>

      {/* Registration Form */}
      <Card>
        <Kicker>Account Details</Kicker>

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
            placeholder="Password (min 6 characters)"
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

        {/* Confirm Password Field */}
        <View style={styles.fieldWrapper}>
          <View style={styles.fieldIcon}>
            <Ionicons name="shield-checkmark" size={20} color={theme.colors.green} />
          </View>
          <TextInput
            value={confirmPassword}
            onChangeText={setConfirmPassword}
            secureTextEntry={!showPassword}
            placeholder="Confirm password"
            placeholderTextColor="#999"
            style={styles.fieldInput}
          />
          {confirmPassword.length > 0 && (
            <View style={styles.matchIndicator}>
              <Ionicons 
                name={password === confirmPassword ? 'checkmark-circle' : 'close-circle'} 
                size={20} 
                color={password === confirmPassword ? theme.colors.green : theme.colors.danger} 
              />
            </View>
          )}
        </View>
      </Card>

      {/* Role Selection */}
      <Card>
        <Kicker>Account Type</Kicker>
        <Body style={{ marginBottom: 12 }}>Select the type of account you want to create:</Body>

        <View style={styles.roleRow}>
          <Pressable 
            style={[styles.roleBtn, role === 'user' && styles.roleBtnActive]}
            onPress={() => setRole('user')}
          >
            <View style={[styles.roleIcon, role === 'user' && styles.roleIconActive]}>
              <Ionicons name="person" size={24} color={role === 'user' ? '#fff' : theme.colors.green} />
            </View>
            <Text style={[styles.roleLabel, role === 'user' && styles.roleLabelActive]}>User</Text>
            <Body style={styles.roleDesc}>Save predictions to cloud</Body>
          </Pressable>

          <Pressable 
            style={[styles.roleBtn, role === 'admin' && styles.roleBtnActiveAdmin]}
            onPress={() => setRole('admin')}
          >
            <View style={[styles.roleIcon, role === 'admin' && styles.roleIconActiveAdmin]}>
              <Ionicons name="shield" size={24} color={role === 'admin' ? '#fff' : theme.colors.orange} />
            </View>
            <Text style={[styles.roleLabel, role === 'admin' && styles.roleLabelActiveAdmin]}>Admin</Text>
            <Body style={styles.roleDesc}>Manage all records</Body>
          </Pressable>
        </View>
      </Card>

      {/* Error Message */}
      {error && (
        <View style={styles.errorBox}>
          <Ionicons name="alert-circle" size={18} color={theme.colors.danger} />
          <Text style={styles.errorText}>{error}</Text>
        </View>
      )}

      {/* Submit Button */}
      <Card>
        <Pressable 
          style={[styles.submitBtn, !canSubmit && styles.submitBtnDisabled]} 
          onPress={onRegister}
          disabled={!canSubmit}
        >
          {busy ? (
            <Ionicons name="hourglass" size={20} color="#fff" />
          ) : (
            <Ionicons name="checkmark-circle" size={20} color="#fff" />
          )}
          <Text style={styles.submitBtnText}>
            {busy ? 'Creating account...' : 'Create Account'}
          </Text>
        </Pressable>

        {/* Divider */}
        <View style={styles.dividerRow}>
          <View style={styles.dividerLine} />
          <Text style={styles.dividerText}>ALREADY HAVE AN ACCOUNT?</Text>
          <View style={styles.dividerLine} />
        </View>

        {/* Back to Login */}
        <Pressable 
          style={styles.secondaryBtn} 
          onPress={() => navigation.navigate('Login')}
        >
          <Ionicons name="log-in" size={20} color={theme.colors.green} />
          <Text style={styles.secondaryBtnText}>Back to Login</Text>
        </Pressable>
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
  matchIndicator: {
    width: 48,
    height: 48,
    alignItems: 'center',
    justifyContent: 'center',
  },

  /* Role Selection */
  roleRow: {
    flexDirection: 'row',
    gap: 12,
  },
  roleBtn: {
    flex: 1,
    padding: 16,
    borderRadius: 12,
    borderWidth: 2,
    borderColor: theme.colors.border,
    alignItems: 'center',
    backgroundColor: '#fff',
  },
  roleBtnActive: {
    borderColor: theme.colors.green,
    backgroundColor: '#e8f5e9',
  },
  roleBtnActiveAdmin: {
    borderColor: theme.colors.orange,
    backgroundColor: '#fff8e1',
  },
  roleIcon: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: '#e8f5e9',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 8,
  },
  roleIconActive: {
    backgroundColor: theme.colors.green,
  },
  roleIconActiveAdmin: {
    backgroundColor: theme.colors.orange,
  },
  roleLabel: {
    fontSize: 16,
    fontWeight: '700',
    color: theme.colors.text,
    marginBottom: 4,
  },
  roleLabelActive: {
    color: theme.colors.greenDark,
  },
  roleLabelActiveAdmin: {
    color: '#e65100',
  },
  roleDesc: {
    fontSize: 12,
    textAlign: 'center',
    color: theme.colors.muted,
  },

  /* Error */
  errorBox: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    backgroundColor: '#ffebee',
    padding: 14,
    borderRadius: 10,
    marginBottom: 16,
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
    paddingHorizontal: 12,
    color: theme.colors.muted,
    fontSize: 10,
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
});

import React, { memo } from 'react';
import { Platform, SafeAreaView, ScrollView, StyleSheet, View } from 'react-native';
import WebChrome from './WebChrome';
import { theme } from '../theme/theme';

const Screen = memo(function Screen({ children, scroll = true, contentContainerStyle }) {
  const Container = scroll ? ScrollView : View;

  if (Platform.OS === 'web') {
    // On web, WebChrome handles scrolling, so we just use a View container
    return (
      <WebChrome>
        <View style={[styles.webContent, contentContainerStyle]}>
          {children}
        </View>
      </WebChrome>
    );
  }

  return (
    <SafeAreaView style={styles.safeArea}>
      <Container
        style={styles.container}
        contentContainerStyle={[styles.content, contentContainerStyle]}
        keyboardShouldPersistTaps="handled"
      >
        {children}
      </Container>
    </SafeAreaView>
  );
});

export default Screen;

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: theme.colors.pageBg,
  },
  container: {
    flex: 1,
  },
  content: {
    flexGrow: 1,
    paddingHorizontal: 16,
    paddingTop: 12,
    paddingBottom: 24,
    width: '100%',
    maxWidth: 980,
    alignSelf: 'center',
  },
  webContent: {
    flexGrow: 1,
    paddingBottom: 12,
  },
});

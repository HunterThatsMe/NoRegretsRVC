export interface FreeSampleItem {
  id: string;
  title: string;
  fileName: string;
  url: string;
  gender: 'male' | 'female';
  style: string;
  durationLabel: string;
  transcript: string;
}

export const FREE_SAMPLES: FreeSampleItem[] = [
  {
    id: 'male-speaking-1',
    title: 'Male Speaking 1',
    fileName: 'MaleSpeakingTest-1.wav',
    url: '/assets/aistudio/MaleSpeakingTest-1_931b5ffd.wav',
    gender: 'male',
    style: 'Expressive Narration',
    durationLabel: '~9.5s',
    transcript:
      'If you need an expressive voice for comedic gaming, animation, weird science explainers, or just any other casual narration, this is the right voice.',
  },
  {
    id: 'male-speaking-2',
    title: 'Male Speaking 2',
    fileName: 'MaleSpeakingTest-2.wav',
    url: '/assets/aistudio/MaleSpeakingTest-2_c2e6e7fe.wav',
    gender: 'male',
    style: 'Conversational Voiceover',
    durationLabel: '~6.5s',
    transcript:
      'Need a great AI voice? Look no further than this conversational clone trained by a professional voice actor.',
  },
  {
    id: 'male-speaking-3',
    title: 'Male Speaking 3',
    fileName: 'MaleSpeakingTest-3.wav',
    url: '/assets/aistudio/MaleSpeakingTest-3_d78e8233.wav',
    gender: 'male',
    style: 'Casual Dialogue',
    durationLabel: '~6.6s',
    transcript:
      "Sure, I can get you in this week. I've got Thursday at 10 or Friday at 2:30. Which one works better for you?",
  },
  {
    id: 'female-speaking-1',
    title: 'Female Speaking 1',
    fileName: 'FemaleSpeakingTest-1.wav',
    url: '/assets/aistudio/FemaleSpeakingTest-1_bb3d674f.wav',
    gender: 'female',
    style: 'Calm Professional',
    durationLabel: '~7.1s',
    transcript:
      "Hi there, I'm here to help with questions, support requests, and quick answers. All delivered with a calm, friendly, professional touch.",
  },
  {
    id: 'female-speaking-2',
    title: 'Female Speaking 2',
    fileName: 'FemaleSpeakingTest-2.wav',
    url: '/assets/aistudio/FemaleSpeakingTest-2_d1823132.wav',
    gender: 'female',
    style: 'Clear Narration',
    durationLabel: '~7.1s',
    transcript:
      'By repeating what students say, teachers can demonstrate that they are listening. By extending what students say...',
  },
  {
    id: 'female-speaking-3',
    title: 'Female Speaking 3',
    fileName: 'FemaleSpeakingTest-3.wav',
    url: '/assets/aistudio/FemaleSpeakingTest-3_458bd4fc.wav',
    gender: 'female',
    style: 'Friendly Assistant',
    durationLabel: '~7.1s',
    transcript:
      "I can help with appointments, reminders, and general questions. If you need a person, I'll make sure you reach the right team.",
  },
];

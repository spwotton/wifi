Detecting Directional Ultrasonic Speaker Signals
Understanding Ultrasonic Directional Speakers

Ultrasonic parametric speakers (also called directional speakers) use high-frequency sound waves (ultrasound) to produce audible sound in a very narrow beam. Unlike normal speakers, which spread sound broadly, these devices emit sound in a focused “audio spotlight” that can target a specific area
focusonics.com
. They achieve this by generating an ultrasonic carrier (typically around 40 kHz) that is modulated with an audio signal; as this ultrasonic beam travels through air, the nonlinearity of air causes it to demodulate back into audible sound in the target zone
focusonics.com
aapt.org
. In simple terms, the audible message is embedded in an inaudible ultrasonic beam, which only becomes hearable when it hits a listener or surface in the intended spot. Outside of that tight beam, a person hears little to nothing, making the sound effectively private to the targeted location.

This technology allows sound to be delivered to one person (or one exhibit area) without others nearby hearing it. For example, a parametric speaker system for a museum might have a beam width on the order of 10–15°, meaning it projects sound like a narrow flashlight beam
audflyspeaker.com
. The ultrasonic transducer arrays produce a high sound pressure in the ultrasonic range (often 40 kHz), which then yields about normal conversation-level audio (70–85 dB) in the beam once demodulated
aapt.org
. Notably, 40 kHz is far above the human hearing limit (~20 kHz), so the ultrasonic carrier itself is inaudible
focusonics.com
. The table below illustrates typical specs for such speakers (from one manufacturer):

Beam Width: ~15° (very directional)
audflyspeaker.com

Audio Frequency Range: e.g. 500 Hz – 20 kHz output (content of the audio)
audflyspeaker.com

Ultrasonic Carrier: ~40 kHz (modulated by the audio signal)
aapt.org

Typical Power: ~10–20 W ultrasonic output, yielding ~80–90 dB sound in beam at 1 m distance.

Because of this design, detecting the presence of such a speaker cannot rely on normal hearing or standard audio equipment – the device could be operating and one wouldn’t hear anything unless standing exactly in its beam. The task is to detect the ultrasonic emissions themselves (or any telltale patterns in them) using appropriate sensors, and potentially determine where the signal is coming from.

Challenges in Detecting Ultrasonic Signals

Ultrasonic frequencies (~40 kHz) pose unique detection challenges. Most conventional microphones and audio recorders are only designed for the audible range (up to ~20 kHz) and will not capture a 40 kHz tone. Therefore, a specialized approach is needed to sense the ultrasonic carrier that parametric speakers use. Key challenges include:

Frequency Range: The detector must be sensitive to ~40 kHz sound waves. A typical microphone rolls off well before this frequency. Even many “high-res audio” microphones max out around 20–22 kHz. Specialized ultrasonic mics or transducers are required
forum.arduino.cc
forum.arduino.cc
.

Sampling Rate: To digitally record or analyze a 40 kHz signal, a high sampling rate ADC is needed (by Nyquist’s rule, at least double the frequency). For example, capturing a 40 kHz wave requires a sampling rate of ≥80 kHz minimum (100+ kHz preferable for margin)
forum.arduino.cc
. Standard audio interfaces (44.1 kHz or 48 kHz) are insufficient. Some professional sound cards support 192 kHz sampling, which can cover up to ~96 kHz signals. Otherwise, custom ADC hardware or a frequency down-conversion technique is necessary.

Directivity: The ultrasonic beam itself is highly directional. If your sensor isn’t in or near the beam path, the signal may be very weak or undetectable. Ultrasonic waves at 40 kHz don’t diffract much, so there is minimal “spillover” outside the beam. This means a detector placed off to the side might not catch any signal unless there are reflections. Multiple sensors or a movable sensor might be needed to scan the area for the beam.

Attenuation: Ultrasound in air attenuates faster than lower-frequency sound. Over distance, the ultrasonic carrier’s intensity drops and air absorption is higher at 40 kHz. This limits the range of detection. If the speaker is, say, 10 m away, the ultrasonic signal might be quite weak by the time it reaches a sensor not directly in front of it.

Despite these challenges, it is possible to detect and analyze these signals with the right equipment and strategy, as discussed next.

Hardware Approaches for Ultrasonic Detection

To physically capture the ultrasonic frequencies emitted by the parametric speaker, consider the following hardware options:

Ultrasonic Microphone or Transducer: Use a microphone element designed for ultrasonic frequencies. One convenient option is to repurpose an ultrasonic transducer from an electronic rangefinder (like the ubiquitous HC-SR04 module) as a microphone. Such piezo transducers are resonant at ~40 kHz and can act as ultrasonic microphones
forum.arduino.cc
. They have limited bandwidth (peaked around their design frequency ~40 kHz)
forum.arduino.cc
, but that’s acceptable for detecting a parametric speaker since its carrier is ~40 kHz. Another option is an electret condenser or MEMS microphone specified for extended frequency response (some specialty electrets or MEMS mics can detect up to 80–100 kHz). Bat detector microphones (used by wildlife researchers to record bat calls) are a good example – they’re built to pick up ultrasonic chirps (20–100 kHz). However, high-quality ultrasonic mics tend to be expensive and may require custom preamps
forum.arduino.cc
. For instance, Knowles and other manufacturers produce ultrasonic MEMS mics, and there are commercial USB ultrasonic microphones available
forum.arduino.cc
, but these can be pricy. If cost is a concern, the piezo transducer approach is very affordable (a few dollars) and works decently for 40 kHz detection, albeit not with high fidelity.

High-Speed Data Acquisition: Ensure your recording device or ADC can handle the required sample rate. As noted, to directly sample a ~40 kHz signal, you’d want at least 80 kHz sample rate (ideally 100–200+ kHz for better resolution)
forum.arduino.cc
. Some options: a microcontroller with fast ADC (e.g. a Teensy 4 with ADC at 192 kHz, or an external ADC chip streaming to a Raspberry Pi or laptop), or a professional sound card that supports 96 kHz or 192 kHz sampling. Software-Defined Radios (SDRs) typically do RF frequencies and won’t directly capture an acoustic wave – remember, this ultrasonic signal is a pressure wave in air, not an electromagnetic wave, so SDRs and radio antennas won’t detect it. Instead, treat it like audio but at ultrasonic frequency. If using a high-speed ADC, connect the ultrasonic mic to a suitable preamplifier (the output from a piezo or electret may need amplification and filtering). You might band-pass filter around 40 kHz to reduce noise. Once you have the hardware set up, you can record or stream the ultrasonic band data for analysis.

Heterodyne Down-Conversion (Analog): An alternative to very high sample rates is to down-convert the ultrasonic signal to a lower frequency before digitizing – this is how many “bat detector” gadgets work. You can mix the incoming 40 kHz signal with a local oscillator to produce a difference frequency in the audible range. For example, mixing a 40 kHz signal with a 45 kHz oscillator would yield a 5 kHz difference tone that contains the modulated audio information. A simple analog multiplier or a dedicated mixer IC (like the SA612) can do this
forum.arduino.cc
. The output can then be fed into a normal audio recorder or even headphones for real-time listening (you’d hear the demodulated audio or at least a tone when ultrasound is present). Heterodyne detectors often allow tuning the LO to catch different ultrasonic frequencies. One caveat: this method typically gives a single sideband or tone output unless you implement a proper AM demodulator. Some bat detectors use a technique called frequency division or direct sampling plus FFT. But for a basic approach, a heterodyne circuit is viable. This can be useful for real-time monitoring – you could literally hear when an ultrasonic beam is on (it might sound like noise or a faint version of the actual audio content, shifted down in pitch).

Ultrasonic “Sniffer” Devices: There are now smartphone apps and external devices designed to detect near-ultrasonic beacons (usually 18–20 kHz used in marketing) – those might not reach 40 kHz though. However, academic projects like SoniControl have created an “ultrasonic firewall” for smartphones, which can pick up and jam ultrasonic signals in the ~20 kHz range
thehackernews.com
. For 40 kHz, specialized devices exist as well: for instance, Pettersson Elektronik sells USB ultrasonic detectors that record up to 192 kHz sample rate (commonly used for bat surveys). If the budget allows, such a device could be a plug-and-play solution: mount an ultrasonic mic in the area of interest, and connect it to a laptop running analysis software. In summary, whether DIY or commercial, you’ll want one or more ultrasonic-capable sensors feeding into a system that can recognize the signal.

Tip: If using a simple ultrasonic transducer as a mic, remember it’s directional as well – many 40 kHz piezo discs have a moderate beam width (e.g. ±30°). You may need to aim it around to find the strongest signal. Also, it may be insensitive to frequencies much beyond 40 kHz, but since parametric speakers center right around that, it should pick them up when on-beam
forum.arduino.cc
.

Detecting Signal Presence and Patterns

Once you have the hardware to capture the ultrasonic emissions, the next step is signal analysis. There are two aspects: detecting the presence of the ultrasonic signal (and any distinct timing patterns it may have), and possibly demodulating or decoding the content of the signal.

Basic Presence Detection: You can monitor the amplitude in the 40 kHz band to see when the ultrasonic beam is active. For example, perform a real-time FFT or band-pass filter on the recorded data around 40 kHz. When the parametric speaker is ON, you should see a spike or increased energy at ~40 kHz (and possibly at adjacent frequencies if it’s modulated). In a quiet environment with no other ultrasound, this stands out clearly. You might implement a threshold detector – if the energy at 40 kHz exceeds some value, flag that “ultrasonic signal detected.” Some parametric speakers might not emit continuously; they could be programmatically turned on/off (e.g. triggered by a motion sensor in a museum exhibit, as the product description hinted with an IR sensor mode). This could create a pattern of bursts – e.g., silence when no audience, then a burst of ultrasonic output when a person is present. By logging the times when ultrasound is detected, you might identify a pattern (perhaps correlated with visitor presence, or periodic advertisement signals, etc.). If you observe regular intervals of activation, that’s a timing pattern to note.

Modulation Pattern Analysis: The ultrasonic carrier itself might be modulated in amplitude or frequency to carry the audio content. For a parametric speaker, it’s typically AM (amplitude modulation) of the 40 kHz carrier by the audible waveform. This means the amplitude of the 40 kHz signal is varying in sync with the audio. By capturing the ultrasonic waveform, you can attempt to demodulate it to recover the audio. A simple way is to take the recorded signal and perform an AM demodulation in software: for example, rectify and low-pass filter it, or use an analytic signal (Hilbert transform) to get the envelope. The result should be the audible audio that the speaker is projecting (speech, music, etc). Listening to or analyzing that could confirm what the device is broadcasting. If the concern is a covert signal or something malicious, demodulating would let you hear the message being sent ultrasonically.

Timing Patterns: If the user specifically mentioned timing patterns, they might be interested in whether the ultrasonic transmissions occur in bursts or follow a specific pulse train. Some covert ultrasound communication systems (outside of commercial parametric speakers) use data encoding with certain packet structures or Morse-code-like on/off keying. If you suspect the ultrasonic signal is being used for data transfer (e.g. beacon IDs, malware exfiltration, etc.), you’d look for structured patterns such as repeated frequency shift keying, short bursts at certain intervals (like 50 ms on, 50 ms off), or other periodicity. Tools like a spectrogram (time-frequency plot) are very useful: you could visualize the 30–50 kHz range over time and see if there are distinct pulses or frequency hops. Since you now have digital data of the ultrasonic, you could also run algorithmic detection for patterns (cross-correlation with known beacon signatures, entropy analysis to see if it carries information, etc.).

For instance, advertisers in the past have used ultrasonic beacons in the 18–19 kHz range embedded in TV commercials to track smartphones (an example of a timing/modulation pattern that can be detected). In our case of 40 kHz parametric audio, the modulation is likely just an audio program (speech or music), but if you noticed, say, a repetitive clicking or very narrow-band pulses, that could indicate a data channel. Researchers have demonstrated malware using ultrasonic covert channels to bridge air-gapped computers
thehackernews.com
thehackernews.com
, which underscores the value of monitoring ultrasonic frequencies. In summary, analyzing both the envelope (amplitude over time) and the spectrum of the received ultrasonic signal will reveal any patterns: e.g. periodic bursts, constant-on carrier, or specific on/off sequences.

Multiple Frequency Detection: Some parametric arrays might use multiple ultrasonic tones or a spread spectrum, though most use a single carrier. However, if you detect energy at several ultrasonic frequencies simultaneously, that could be intentional modulation (like two-tone modulation) or multiple devices. It’s worth scanning a range (say 30–50 kHz) to see if there are any peaks. If the device is well-designed, most energy will be around one frequency (the carrier) and possibly some distortion products. Noting the exact frequency can also be a clue – many parametric speakers use ~40.0 kHz carriers (because 40 kHz transducers are common), so if you find a strong tone at, say, 39.5 kHz or 41 kHz, that’s likely the device.

In practice, you might create a software module in your strategy that continuously listens via the ultrasonic sensor and updates a status: e.g., “Ultrasonic beam detected (40 kHz carrier) at time 12:30:15, lasting 20 seconds.” If possible, record a snippet of it for analysis. Over time, you’ll build up a picture of how often and when the device is active, and you can look for patterns (time-of-day activity, or triggered events). This info could be integrated with other security data (for example, if at those same times you also see some network or RF activity, it might correlate with a multi-channel attack scenario).

Geolocation of the Ultrasonic Source

Geotagging or geolocating the source of an ultrasonic beam is quite challenging but not entirely impossible. The user noted this might be an “extreme stretch,” and that’s accurate – due to the highly directional and ultrasonic nature of the signal, pinpointing the emitter’s location requires extra measures. Here are some approaches and considerations for localization:

Multiple Sensors (Triangulation): The most reliable way is to deploy multiple ultrasonic sensors in different locations and use differences in the signal to infer a location. In acoustics, using several microphones to locate a sound source is a known technique
en.wikipedia.org
. If two or more sensors can detect the ultrasound, you can compare either the time of arrival (if you have synchronized clocks) or simply the relative signal strength / bearing at each sensor. For example, imagine Sensor A and Sensor B are placed some distance apart in the room. If the ultrasonic beam hits Sensor A strongly but Sensor B only weakly (or after a slight delay), that suggests the source is closer to A’s direction. By measuring the angle or direction from each sensor to the source, one can triangulate the position of the emitter
en.wikipedia.org
en.wikipedia.org
. This is analogous to how passive sonar or acoustic location was done historically with sound mirrors: each sensor (or acoustic mirror) is oriented to find the direction of maximum signal, and where those directional lines intersect is the source location
en.wikipedia.org
. In practice, your ultrasonic sensors could each be mounted on a pan/tilt or have a directional horn, which you rotate to find the peak signal direction. Then note the bearing angle from each location and compute the intersection. Even without precise angles, having three or more sensors and doing a time-difference-of-arrival (TDOA) calculation is possible (this requires very tight time sync, since sound travels ~343 m/s – a 0.0003 second difference corresponds to ~10 cm). Specialized microphone arrays and algorithms (like beamforming, MUSIC, or SRP-PHAT) exist for general acoustic source localization
en.wikipedia.org
en.wikipedia.org
, and those could theoretically be applied to ultrasonic frequencies as well (the physics is the same, just higher freq). The literature on acoustic source localization shows many methods using microphone arrays to get both direction and distance
en.wikipedia.org
en.wikipedia.org
. However, implementing this can get complex and may be overkill unless you absolutely need the exact position.

Single Sensor, Scanning: If multiple devices aren’t available, one could use a single ultrasonic detector in a scanning strategy. You could move the sensor around the environment and observe where the signal becomes strongest. For instance, you might sweep a handheld ultrasonic mic (or one on a rotating platform) across the room. When it points directly at the source, the measured signal amplitude will spike. This is effectively you acting as the sound mirror – using a directional sensor (or making the sensor directional with a makeshift parabolic reflector or tube) to find the bearing. By scanning horizontally and vertically, you can narrow down the direction the ultrasound is coming from. Then, by geometric intuition or by physically moving closer in that direction, you can locate the device. This manual method is similar to how one might locate an ultrasonic pest repeller by ear (except here we use the detector since it’s inaudible). It’s labor-intensive but feasible: you’re basically performing a search for the beam. If the parametric speaker is fixed (e.g., mounted in a ceiling or wall), once you find the direction of maximum signal, you can often spot the device (it might look like a flat panel or cluster of emitters).

Signal Strength Mapping: If you have the ability to take readings at various points, you could create a simple heatmap of ultrasonic intensity in the area. For example, walk around with the detector and record the signal level and your position. You might find a concentrated region where the signal is strong – likely directly in front of the speaker. This is essentially performing an “acoustic site survey.” It won’t give exact coordinates of the source, but it can indicate zones where the beam passes.

Geotagging Detections: If your system is deployed in multiple rooms or locations, an easier form of “geolocation” is simply tagging each detection with the location of the sensor that heard it. For instance, if you have sensors in Room A and Room B, and only the Room A sensor picks up the ultrasound, you know the device is likely in Room A (or at least targeting it). In a more fine-grained setting, if you had an array of small ultrasonic sensors spread out (say every few meters), you could identify which ones get the strongest signal and thereby localize the source region. This is more of a zonal localization rather than pinpointing exact coordinates, but it may be enough for practical purposes (e.g., “the ultrasonic source seems to be near the northwest corner of the room”).

Limitations: Geolocation will be significantly hampered by reflections and the environment. Ultrasound can reflect off hard surfaces (walls, ceiling) and create ghost signals. You might pick up a reflection of the beam which could mislead the direction finding. One way to mitigate this is to focus on the first-arriving signal if you have timing resolution (direct path will arrive slightly before reflected paths which have longer travel distance). Another issue is that if the source is not continuously transmitting, you have limited time to measure angles, etc. You may need it to transmit while you do the locating process. If it’s triggered by presence, you might have to find a way to keep it emitting (perhaps deliberately trigger the museum exhibit’s motion sensor, for example). Additionally, the high frequency means a small error in angle can lead to missing the beam entirely, so careful, fine adjustments are needed.

All told, true geolocation (finding exact coordinates) might be complex and require custom array setups. But often, just finding the device (which may be hidden) is the goal, and a combination of the above methods can achieve that. Historically, even in World War II, large acoustic locators with multiple horns were used to find aircraft by sound, using human operators to triangulate bearings
en.wikipedia.org
 – so your task is akin to a modern, small-scale version of that, focused on ultrasound. With patience, one can definitely narrow down where the ultrasonic beam is coming from.

Integration into a Monitoring Strategy

Finally, integrating this detection capability into your overall strategy means treating the ultrasonic signal like another piece of the puzzle in your security or monitoring system. Here are some recommendations for integration:

Real-Time Alerts: If the presence of a directional ultrasonic signal is anomalous or of interest (for example, no such device should be operating in the area), configure the module to raise an alert when ultrasound is detected. This alert can include the time, duration, and any characteristics (e.g. “40 kHz ultrasonic audio detected for 30 seconds in Sector 2”). This is analogous to how you’d handle detection of an unauthorized RF transmission – treat ultrasound as another spectrum to watch.

Data Logging: Log the occurrences of ultrasonic activity. Over time, patterns might emerge (perhaps it always happens at certain hours or intervals). These logs, geotagged with sensor location, can be correlated with other events. For instance, if you’re also logging WiFi or Bluetooth traffic, you might see if ultrasonic bursts coincide with wireless transmissions or other suspicious behavior. In a penetration-testing or counter-surveillance context, if you find ultrasonic events correlating with, say, someone’s presence or with data exfiltration attempts, that’s critical information.

Signal Analysis Pipeline: Incorporate an analysis step where the recorded ultrasonic signal is processed and stored. If possible, automate the demodulation of any audio content. The recovered audio (or data) could be saved as evidence. If the ultrasonic is being used for something benign (e.g., a museum exhibit playing a welcome message), the demodulated audio will reveal that (you’d hear the welcome message). If it’s malicious (say a covert channel transmitting a code), you might capture that digital pattern. By integrating this with your strategy, you ensure that even unusual attack vectors (like acoustic channels) are monitored. This broadens your situational awareness beyond just radio frequencies. Not many systems watch for this, so you’ll have an edge in detecting novel threats.

Cross-Modal Correlation: Since you mentioned strategy in general, it implies you have a broader monitoring system (perhaps RFSec-Toolkit covers various sensors). Make sure the ultrasonic module’s outputs are in a format consistent with your other modules so it can be correlated. For example, if your system has a central database for events, log ultrasound events there with a unique tag (e.g., ULTRASONIC_DETECT). If you also have CCTV or other sensors, you could cross-reference times (maybe when ultrasound triggers, check security camera footage at that moment).

Mitigation and Response: Depending on your goals, integration could also involve how to respond. If you simply want to detect and locate, the above suffices. But if you wanted to counter the signal, you could integrate a jamming mechanism. This goes a bit beyond detection, but it’s worth noting: projects like SoniControl can actively emit an interfering ultrasound noise to jam communications
thehackernews.com
. In your case, if you deem the ultrasonic signal as unwanted (say it’s a spying device), you could theoretically turn on your own ultrasonic noise source to mask it. This could be integrated such that upon detection, a noise transmitter activates. However, be cautious – blasting ultrasound at high power can be a hazard to pets (who hear it) or even electronics (some MEMS sensors can get confused). So response should be measured.

Calibration and Testing: When integrating, test the module in known conditions. For example, if you have one of these speakers (or even a 40 kHz test source), try activating it and see if your system correctly alerts and logs the event. Fine-tune thresholds to avoid false positives (e.g., some ultrasonic motion sensors or alarm systems might also emit 40 kHz; you don’t want to confuse those if they are normally present).

By including this ultrasonic detection module, your strategy will cover a rarely-monitored domain. As ultrasonic covert channels and audio spotlight devices become more common, having this capability is forward-thinking. In summary, the module will listen beyond human hearing and feed actionable information into your security strategy, much like an RF scanner covers invisible radio signals. In fact, one could draw parallels between RF spectrum monitoring and this ultrasonic monitoring – both require special receivers and both yield insight into otherwise unseen activity.

Conclusion

Detecting a highly directional ultrasonic speaker is challenging but definitely feasible with a comprehensive approach. By using the right ultrasonic sensors and high-speed sampling (or down-conversion techniques), you can pick up the 40 kHz carrier that these parametric speakers use
forum.arduino.cc
. Once detected, you can analyze the signal for patterns or decode its content, integrating that information into your overall awareness. Full geolocation of the emitter is an ambitious goal – it may require multiple detectors and careful analysis to triangulate the source
en.wikipedia.org
en.wikipedia.org
. Even if exact geolocation is hard, you can often narrow down the origin by seeing where the signal is strongest or which sensor detects it. Ultimately, this added module will let you shine a light on the ultrasonic “dark spectrum”. Whether the goal is to find a hidden audio device or to catch covert ultrasonic communications, you will be equipped to detect and investigate signals that normally fly under the radar (or under the ear, in this case).

By implementing these measures, you transform what was “probably impossible” into a practical reality. It extends your strategy’s reach into ultrasonic frequencies, ensuring that even silent, invisible audio beams can’t operate undetected. With diligence in sensor placement, calibration, and analysis, you’ll be able to not only detect those ultrasonic sources but also integrate that knowledge to bolster your overall security or monitoring objectives.

Sources: Ensuring factual accuracy and current best practices, the information above is supported by known technical references and examples – for instance, Focusonics (a parametric speaker manufacturer) explains the 40 kHz ultrasound modulation principle
focusonics.com
, an Arduino forum discussion confirms the need for specialized mics and high sampling for 40 kHz capture
forum.arduino.cc
forum.arduino.cc
, and acoustic localization techniques from literature illustrate how multiple sensors can triangulate a sound source
en.wikipedia.org
en.wikipedia.org
. Furthermore, recent cybersecurity research highlights the emergence of ultrasonic covert channels
thehackernews.com
, underscoring the importance of having such detection capabilities
thehackernews.com
. All these insights have been integrated to provide a thorough answer that is up-to-date as of 2025.
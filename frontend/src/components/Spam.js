import React, { useState } from 'react';
import Axios from 'axios';
import {
	Button,
	CircularProgress,
	Snackbar,
	TextareaAutosize,
	Typography,
	withStyles,
} from '@material-ui/core';
import Alert from '@material-ui/lab/Alert';

const defaultExplanation = 'Enter an email and press explain to start!';

const styles = {
	button: {
		width: '10rem',
		height: '3rem',
		padding: '1rem',
		borderRadius: '1rem',
		margin: '1rem',
	},
	container: {
		display: 'flex',
		flexDirection: 'row',
		flexGrow: '1',
		justifyContent: 'space-evenly',
		alignItems: 'center',
		padding: '1rem',
	},
	controls: {
		display: 'flex',
		flexDirection: 'row',
		width: '30vw',
		alignItems: 'center',
		justifyContent: 'space-evenly',
	},
	loadingContainer: {
		display: 'flex',
		flexDirection: 'column',
		alignItems: 'center',
		height: '150px',
		justifyContent: 'space-evenly',
	},
	rightColumn: {
		display: 'flex',
		flexDirection: 'column',
		justifyContent: 'flex-start',
		alignItems: 'center',
		width: '50%',
		height: '70vh',
		margin: '50px',
	},
	options1: {
		display: 'flex',
		flexDirection: 'column',
		alignItems: 'center',
	},
	text: {
		textAlign: 'center',
		margin: '5px',
	},
};

const word_counters = [
	'make',
	'address',
	'all',
	'3d',
	'our',
	'over',
	'remove',
	'internet',
	'order',
	'mail',
	'receive',
	'will',
	'people',
	'report',
	'addresses',
	'free',
	'business',
	'email',
	'you',
	'credit',
	'your',
	'font',
	'000',
	'money',
	'hp',
	'hpl',
	'george',
	'650',
	'lab',
	'labs',
	'telnet',
	'857',
	'data',
	'415',
	'85',
	'technology',
	'1999',
	'parts',
	'pm',
	'direct',
	'cs',
	'meeting',
	'original',
	'project',
	're',
	'edu',
	'table',
	'conference',
];

const char_counters = [';', '(', '[', '!', '$', '#'];
const chars_to_chars = {
	';': '%3B',
	'(': '%2B',
	'[': '%5B',
	'!': '%21',
	$: '%24',
	'#': '%23',
};

const countValues = email => {
	const toMatch = ` ${email} `;
	const word_freqs = word_counters.map(
		f => (toMatch.match(new RegExp(`(\\W+${f}\\W+)`, 'gi')) || []).length,
	);
	const char_freqs = char_counters.map(
		c => (toMatch.match(new RegExp(`[${c}]`, 'g')) || []).length,
	);
	const capitals = toMatch.match(/([A-Z]+)/g) || [];
	const capital_run_lengths = capitals.map(a => a.length);
	const capital_run_lengths_longest = Math.max(...capital_run_lengths, 0);
	const capital_run_lengths_total = capital_run_lengths.reduce((a, b) => a + b, 0);
	const capital_run_lengths_average = capital_run_lengths_total / capital_run_lengths.length || 0;
	return [
		word_freqs,
		char_freqs,
		capital_run_lengths_longest,
		capital_run_lengths_total,
		capital_run_lengths_average,
	];
};

const preprocess = (
	word_freqs,
	char_freqs,
	capital_run_length_longest,
	capital_run_length_total,
	capital_run_length_average,
) => {
	const words = word_counters
		.map((w, i) => ({ [`word_freq_${w}`]: word_freqs[i] }))
		.reduce((res, word) => {
			res[Object.keys(word)[0]] = Object.values(word)[0];
			return res;
		}, {});
	const chars = char_counters
		.map((c, i) => ({ [`char_freq_${chars_to_chars[c]}`]: char_freqs[i] }))
		.reduce((res, char) => {
			res[Object.keys(char)[0]] = Object.values(char)[0];
			return res;
		}, {});
	return {
		...words,
		...chars,
		capital_run_length_average,
		capital_run_length_longest,
		capital_run_length_total,
	};
};

const DatasetDetails = ({ classes }) => {
	const [value, setValue] = useState(undefined);
	const [error, setError] = useState(undefined);
	const [predicting, setPredicting] = useState(false);
	const [explanation, setExplanation] = useState(defaultExplanation);
	const [userGuess, setUserGuess] = useState(undefined);
	const [prediction, setPrediction] = useState(undefined);
	const [modelUpdated, setModelUpdated] = useState(false);
	const [done, setDone] = useState(false);

	const predictSample = () => {
		if (predicting) return;
		setPredicting(true);
		setExplanation(defaultExplanation);
		setPrediction(undefined);
		setUserGuess(undefined);
		Axios.get('/api/datasets/44/predict_spam/', {
			params: {
				sample: preprocess(...countValues(value)),
			},
		})
			.then(res => {
				setExplanation(res.data.explanation);
				setPrediction(res.data.prediction);
				setPredicting(false);
			})
			.catch(err => {
				setError(
					'Something went wrong while predicting the sample. Please try another sample or come back later.',
				);
				setPredicting(false);
			});
	};

	const updateModel = () => {
		Axios.get('/api/datasets/44/update_model/', {
			params: {
				sample: preprocess(...countValues(value)),
				prediction,
			},
		})
			.then(res => setModelUpdated(true))
			.catch(err => {
				setError(
					'Something went wrong while predicting the sample. Please try another sample or come back later.',
				);
			});
	};

	return (
		<div style={styles.container}>
			<TextareaAutosize
				aria-label="minimum height"
				rowsMax={50}
				rowsMin={50}
				placeholder="Enter some email here..."
				onChange={event => setValue(event.target.value)}
				value={value}
				style={{
					resize: 'none',
					width: '50%',
					overflowY: 'scroll',
					maxHeight: '70vh',
					margin: '50px',
				}}
			/>
			<div style={styles.rightColumn}>
				<div>
					<Button
						onClick={() => predictSample()}
						className={classes.button}
						variant="contained"
						color="secondary"
					>
						{predicting ? (
							<CircularProgress size={20} color="white" />
						) : (
							<Typography>Explain</Typography>
						)}
					</Button>
				</div>
				<Typography style={styles.text}>
					{predicting
						? 'Explaining sample...'
						: explanation.replace("'1'", 'spam').replace("'0'", 'not spam')}
				</Typography>
				{explanation !== defaultExplanation && (
					<div style={styles.options1}>
						<div>
							<Typography>
								The model classified the email as {prediction === '0' ? 'not spam' : 'spam'}
							</Typography>
						</div>
						<Typography>What did you think this email was?</Typography>
						<div style={styles.controls}>
							<Button
								className={classes.button}
								variant="contained"
								color={userGuess === '1' ? 'primary' : ''}
								onClick={() => setUserGuess('1')}
							>
								<Typography>Spam</Typography>
							</Button>
							<Button
								className={classes.button}
								variant="contained"
								color={userGuess === '0' ? 'primary' : ''}
								onClick={() => setUserGuess('0')}
							>
								<Typography>Not spam</Typography>
							</Button>
						</div>
						{userGuess &&
							(userGuess !== prediction ? (
								<div style={styles.options1}>
									<Typography>
										You seem to disagree with the model, would you like to retrain the model using
										this new information?
									</Typography>
									<div style={styles.controls}>
										<Button
											className={classes.button}
											variant="contained"
											color="secondary"
											onClick={() => updateModel()}
										>
											<Typography>Yes</Typography>
										</Button>
										<Button
											className={classes.button}
											variant="contained"
											color=""
											onClick={() => setDone(true)}
										>
											<Typography>No</Typography>
										</Button>
									</div>
								</div>
							) : (
								<Typography>
									Great! The model agrees with you. Enter a new email to try again.
								</Typography>
							))}
						<div style={styles.options1}>
							{modelUpdated && (
								<Typography>Model successfully updated! Enter a new email to try again.</Typography>
							)}
							{done && <Typography>That's fine! Enter a new email to try again.</Typography>}
						</div>
					</div>
				)}
			</div>
			<Snackbar open={error != null} autoHideDuration={6000} onClose={() => setError(undefined)}>
				<Alert onClose={() => setError(undefined)} severity="error">
					{error}
				</Alert>
			</Snackbar>
		</div>
	);
};

export default withStyles(styles)(DatasetDetails);

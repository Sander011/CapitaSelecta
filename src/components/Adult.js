import React, { useState } from 'react';
import Axios from 'axios';
import {
	Button,
	CircularProgress,
	MenuItem,
	Select,
	Slider,
	Typography,
	withStyles,
} from '@material-ui/core';

const defaultPrediction = 'Enter feature values and press explain to start!';

const styles = {
	container: {
		display: 'flex',
		flexDirection: 'row',
		flexGrow: '1',
		justifyContent: 'space-evenly',
		alignItems: 'center',
	},
	loadingContainer: {
		display: 'flex',
		flexDirection: 'column',
		alignItems: 'center',
		height: '150px',
		justifyContent: 'space-evenly',
	},
	leftColumn: {
		display: 'flex',
		flexDirection: 'column',
		justifyContent: 'flex-start',
		alignItems: 'flex-start',
		width: '50%',
		height: '70vh',
		margin: '50px',
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
	input: {
		display: 'flex',
		width: '100%',
		justifyContent: 'space-between',
		margin: '10px',
	},
	inputText: {
		width: '10vw',
	},
	inputChanger: {
		width: '20vw',
		display: 'flex',
	},
	sliderText: {
		marginLeft: '25px',
		width: '25px',
	},
	options: {
		display: 'flex',
		flexDirection: 'column',
		alignItems: 'center',
	},
	text: {
		textAlign: 'center',
		margin: '5px',
	},
};

const Adult = ({ classes, features, valuesPerFeature, boundsPerFeature, setError, setSamples }) => {
	const [featureValues, setFeatureValues] = useState({});
	const [predicting, setPredicting] = useState(false);
	const [explanation, setExplanation] = useState(undefined);
	const [userGuess, setUserGuess] = useState('');
	const [prediction, setPrediction] = useState(defaultPrediction);
	const [done, setDone] = useState(false);
	const [rules, setRules] = useState([]);
	const [showAllRules, setShowAllRules] = useState(false);
	const [showExplanation, setShowExplanation] = useState(false);
	const [generating, setGenerating] = useState(false);

	const reset = () => {
		setPrediction(defaultPrediction);
		setUserGuess('');
		setRules([]);
		setShowAllRules(false);
		setShowExplanation(false);
		setExplanation(undefined);
		setDone(false);
	};

	const predictSample = () => {
		if (predicting || prediction !== defaultPrediction) return;

		setPredicting(true);
		reset();

		const sampleFeatures = {};

		features.forEach((f) => {
			if (f in valuesPerFeature) sampleFeatures[f] = featureValues[f] || valuesPerFeature[f][0];
			if (f in boundsPerFeature) sampleFeatures[f] = featureValues[f] || boundsPerFeature[f][0];
		});

		Axios.get('/api/datasets/predict_adult/', {
			params: {
				sample: sampleFeatures,
			},
		})
			.then((res) => {
				let explanation = res.data.explanation;
				explanation = explanation.replace('predicted', 'predicted that this adult has an income');
				explanation = explanation.replace(/<=/gi, '≤');
				explanation = explanation.replace(/>=/gi, '≥');
				explanation = explanation.replace(/\/=/gi, '≠');
				explanation = explanation.replace(/'/gi, '');
				const rules = explanation.split('because')[1].split(' and');
				setExplanation(`${explanation.split('because')[0]} because ${rules[0]}`);
				setRules(rules);
				setPrediction(explanation.split(' instead')[0]);
			})
			.catch(() => {
				setError(
					'Something went wrong while predicting the sample. Please try another sample or come back later.',
				);
			})
			.finally(() => setPredicting(false));
	};

	const handleChange = (f, v) => {
		reset();
		setFeatureValues({ ...featureValues, [f]: v });
	};

	const generate_nearby = () => {
		const sampleFeatures = {};
		setGenerating(true);
		features.forEach((f) => {
			if (f in valuesPerFeature) sampleFeatures[f] = featureValues[f] || valuesPerFeature[f][0];
			if (f in boundsPerFeature) sampleFeatures[f] = featureValues[f] || boundsPerFeature[f][0];
		});

		Axios.get('/api/datasets/nearby_samples/', {
			params: {
				sample: sampleFeatures,
			},
		})
			.then((res) => setSamples(res.data.data))
			.catch(() => {
				setError('Something went wrong while updating. Please try again or come back later.');
			});
	};

	const renderSampleGenerator = () =>
		features.length > 0 ? (
			features.map((f) => (
				<div key={f} style={styles.input}>
					<Typography style={styles.inputText}>{f}</Typography>
					{renderValueChangers(f)}
				</div>
			))
		) : (
			<CircularProgress style={{ margin: 'auto' }} size={40} color={'primary'} />
		);

	const renderValueChangers = (f) => {
		if (f in valuesPerFeature) {
			return (
				<Select
					value={featureValues[f] || valuesPerFeature[f][0]}
					onChange={(e) => handleChange(f, e.target.value)}
					style={styles.inputChanger}
				>
					{valuesPerFeature[f].map((v) => (
						<MenuItem key={v} value={v}>
							{v}
						</MenuItem>
					))}
				</Select>
			);
		} else if (f in boundsPerFeature) {
			const min = parseInt(boundsPerFeature[f][0]);
			const max = parseInt(boundsPerFeature[f][1]);
			return (
				<div style={styles.inputChanger}>
					<Slider
						min={min}
						max={max}
						marks={[
							{ value: min, label: min },
							{ value: max, label: max },
						]}
						step={1.0}
						valueLabelDisplay="auto"
						value={featureValues[f] || min}
						onChange={(_, v) => handleChange(f, v)}
					/>
					<Typography style={styles.sliderText}>{featureValues[f] || min}</Typography>
				</div>
			);
		}
	};

	const renderPrediction = () => (
		<React.Fragment>
			<Button
				onClick={() => predictSample()}
				className={classes.button}
				variant="contained"
				color={prediction !== defaultPrediction ? 'primary' : 'secondary'}
			>
				{predicting ? (
					<CircularProgress size={20} color="white" />
				) : (
					<Typography>Predict</Typography>
				)}
			</Button>
			<Typography style={styles.text}>
				{predicting ? 'Predicting sample...' : prediction}
			</Typography>
		</React.Fragment>
	);

	const renderWhyButton = () => (
		<Button
			className={classes.button}
			variant="contained"
			onClick={() => setShowExplanation(true)}
			color={showExplanation ? 'primary' : ''}
		>
			<Typography variant="p">Why {prediction.split('income ')[1]}?</Typography>
		</Button>
	);

	const renderExplanation = () => (
		<React.Fragment>
			<Typography style={styles.text}>{explanation}</Typography>
			<Typography
				onClick={() => setShowAllRules(!showAllRules)}
				style={{ ...styles.text, textDecoration: 'underline', cursor: 'pointer' }}
			>
				{rules.length > 1 ? `show ${showAllRules ? 'less' : 'more'} rules` : ''}
			</Typography>
			<Typography style={{ ...styles.text, whiteSpace: 'pre-wrap' }}>
				{showAllRules && rules.slice(1).join('\n')}
			</Typography>
		</React.Fragment>
	);

	const renderChoice = () => (
		<React.Fragment>
			<Typography style={styles.text}>What did you think the income of this adult was?</Typography>
			<div className={classes.controls}>
				<Button
					className={classes.button}
					variant="contained"
					color={userGuess === '≤50K' ? 'primary' : ''}
					onClick={() => setUserGuess('≤50K')}
					disabled={userGuess === '>50K'}
				>
					<Typography>{'≤50K'}</Typography>
				</Button>
				<Button
					className={classes.button}
					variant="contained"
					color={userGuess === '>50K' ? 'primary' : ''}
					onClick={() => setUserGuess('>50K')}
					disabled={userGuess === '≤50K'}
				>
					<Typography>{'>50K'}</Typography>
				</Button>
			</div>
		</React.Fragment>
	);

	const renderUserGuess = () => (
		<div style={styles.options}>
			<Typography style={styles.text}>
				You seem to disagree with the model, would you like to walk through some related samples in
				order to improve the model?
			</Typography>
			<div className={classes.controls}>
				<Button
					className={classes.button}
					variant="contained"
					color={generating ? 'primary' : ''}
					onClick={() => generate_nearby()}
					disabled={done}
				>
					{generating ? <CircularProgress size={20} color="white" /> : <Typography>Yes</Typography>}
				</Button>
				<Button
					className={classes.button}
					variant="contained"
					color={done ? 'primary' : ''}
					onClick={() => setDone(true)}
					disabled={generating}
				>
					<Typography>No</Typography>
				</Button>
			</div>
		</div>
	);

	return (
		<div style={styles.container}>
			<div style={styles.leftColumn}>{renderSampleGenerator()}</div>
			<div style={styles.rightColumn}>
				{renderPrediction()}
				{prediction && prediction !== defaultPrediction && renderWhyButton()}
				{showExplanation && renderExplanation()}
				{showExplanation && renderChoice()}
				{userGuess && userGuess !== prediction.split('income ')[1] && renderUserGuess()}
				{(done || userGuess === prediction.split('income ')[1]) && (
					<Typography style={styles.text}>Great! Enter new details to try again.</Typography>
				)}
			</div>
		</div>
	);
};

export default withStyles(styles)(Adult);

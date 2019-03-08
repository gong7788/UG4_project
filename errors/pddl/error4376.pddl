(define (problem block-problem)
	(:domain blocksworld)
	(:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 t0)
	(:init 
		(arm-empty )
		(on-table b0)
		(clear b0)
		(on-table b1)
		(clear b1)
		(on-table b2)
		(clear b2)
		(clear b3)
		(on-table b8)
		(clear b8)
		(in-tower t0)
		(on b9 t0)
		(in-tower b9)
		(on b7 b9)
		(in-tower b7)
		(on b6 b7)
		(in-tower b6)
		(on b5 b6)
		(in-tower b5)
		(on b4 b5)
		(in-tower b4)
		(on b3 b4)
		(in-tower b3)
		(green b0)
		(yellow b0)
		(red b0)
		(green b1)
		(yellow b1)
		(blue b1)
		(green b2)
		(yellow b2)
		(green b3)
		(red b3)
		(blue b3)
		(green b4)
		(blue b4)
		(green b5)
		(yellow b5)
		(blue b5)
		(green b6)
		(blue b6)
		(blue b7)
		(green b8)
		(yellow b8)
		(blue b8)
		(green b9)
		(yellow b9)
		(blue b0)
	)
	(:goal (and (forall (?x) (in-tower ?x)) (forall (?y) (or (not (yellow ?y)) (exists (?x) (and (green ?x) (on ?x ?y))))) (forall (?x) (or (not (red ?x)) (exists (?y) (and (blue ?y) (on ?x ?y))))) (and (not (on b8 b9)) (not (on b8 b7)) (not (on b8 b6)) (not (on b8 b5)) (not (on b8 b4)) (not (on b8 b3)))))
)
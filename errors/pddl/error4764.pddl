(define (problem block-problem)
	(:domain blocksworld)
	(:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 t0)
	(:init 
		(arm-empty )
		(clear b2)
		(on-table b3)
		(clear b3)
		(on-table b8)
		(clear b8)
		(in-tower t0)
		(on b6 t0)
		(in-tower b6)
		(on b4 b6)
		(in-tower b4)
		(on b9 b4)
		(in-tower b9)
		(on b7 b9)
		(in-tower b7)
		(on b5 b7)
		(in-tower b5)
		(on b1 b5)
		(in-tower b1)
		(on b0 b1)
		(in-tower b0)
		(on b2 b0)
		(in-tower b2)
		(blue b2)
		(blue b3)
		(red b8)
		(red b3)
		(green b2)
	)
	(:goal (and (forall (?x) (in-tower ?x)) (forall (?y) (or (not (yellow ?y)) (exists (?x) (and (green ?x) (on ?x ?y))))) (forall (?x) (or (not (red ?x)) (exists (?y) (and (blue ?y) (on ?x ?y))))) (forall (?y) (or (not (blue ?y)) (exists (?x) (and (red ?x) (on ?x ?y))))) (and (not (on b8 b4)) (not (on b8 b9)) (not (on b5 b9)) (not (on b8 b7)) (not (on b8 b5)) (not (on b2 b1)) (not (on b3 b2)))))
)
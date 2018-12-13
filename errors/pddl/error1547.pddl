(define (problem block-problem)
	(:domain blocksworld)
	(:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 t0)
	(:init 
		(arm-empty )
		(on-table b3)
		(clear b3)
		(clear b4)
		(on-table b5)
		(clear b5)
		(on-table b6)
		(clear b6)
		(on-table b9)
		(clear b9)
		(in-tower t0)
		(on b8 t0)
		(in-tower b8)
		(on b7 b8)
		(in-tower b7)
		(on b0 b7)
		(in-tower b0)
		(on b1 b0)
		(in-tower b1)
		(on b2 b1)
		(in-tower b2)
		(on b4 b2)
		(in-tower b4)
		(blue b0)
		(red b1)
		(blue b1)
		(red b2)
		(blue b2)
		(red b3)
		(blue b3)
		(blue b4)
		(red b5)
		(blue b5)
		(red b6)
		(blue b6)
		(red b7)
		(blue b7)
		(red b8)
		(blue b8)
		(red b9)
		(blue b9)
	)
	(:goal (and (forall (?x) (in-tower ?x)) (forall (?x) (or (not (red ?x)) (exists (?y) (and (blue ?y) (on ?x ?y))))) (and (not (on b9 t0)) (not (on b9 b8)) (not (on b9 b7)) (not (on b5 b4)))))
)